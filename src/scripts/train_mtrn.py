import argparse
import logging

from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from systems import EpicActionRecogintionShapleyClassifier

from models.esvs import V_MTRN, N_MTRN

from datasets.pickle_dataset import MultiPickleDataset
from frame_sampling import RandomSampler

from ipdb import launch_ipdb_on_exception
from tqdm import tqdm

import plotly.graph_objects as go
import numpy as np
import pickle

from livelossplot import PlotLosses

parser = argparse.ArgumentParser(
    description="Extract per-frame features from given dataset and backbone",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("features_pkl", type=Path, help="Path to pickle file to save features")
parser.add_argument("model_params_dir", type=Path, help="Path to save model parameters (not file name)")
parser.add_argument("--val-features-pkl", type=Path, help="Path to validation features pickle")
parser.add_argument("--min-frames", type=int, default=1, help="min frames to train models for")
parser.add_argument("--max-frames", type=int, default=8, help="max frames to train models for")
parser.add_argument("--batch-size", type=int, default=512, help="mini-batch size of frame features to run through ")
parser.add_argument("--train-test-split", type=float, default=0.3, help="Train test split if no validation features given")
parser.add_argument("--epoch", type=int, default=200, help="How many epochs to do over the dataset")
parser.add_argument("--type", type=str, default='verb', help="Which class to train")
# parser.add_argument("results_pkl", type=Path, help="Path to save training results")
# parser.add_argument("--test", type=bool, default=False, help="Set test mode to true or false on the RandomSampler")
# parser.add_argument("--log_interval", type=int, default=10, help="How many iterations between outputting running loss")
# parser.add_argument("--n_frames", type=int, help="Number of frames for 2D CNN backbone")
# parser.add_argument("--save_fig", type=Path, help="Save a graph showing lr / loss")

def no_collate(args):
    return args

def train_test_loader(dataset: MultiPickleDataset, batch_size: int, val_split: float) -> Tuple[DataLoader, DataLoader]:
    idxs = list(range(len(dataset)))
    split = int(np.floor(val_split * len(dataset)))
    np.random.shuffle(idxs)

    train_idx, test_idx = idxs[split:], idxs[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    return DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=no_collate), DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=no_collate)

def main(args):
    
    device = torch.device("cuda:0")
    dtype = torch.float
    
    if args.type == 'verb':
        models = [V_MTRN(frame_count=i).to(device) for i in range(1,args.max_frames+1)]
        optimisers = [Adam(m.parameters(), lr=3e-4) for m in models]
        frame_samplers = [RandomSampler(frame_count=m.frame_count, snippet_length=1, test=False) for m in models]
    elif args.type == 'noun':
        models = [N_MTRN(frame_count=i).to(device) for i in range(1,args.max_frames+1)]
        optimisers = [Adam(m.parameters(), lr=3e-4) for m in models]
        frame_samplers = [RandomSampler(frame_count=m.frame_count, snippet_length=1, test=False) for m in models]
    else:
        raise ValueError(f"unknown type: {args.type}, known types are 'verb' and 'noun'")

    dataset = MultiPickleDataset(args.features_pkl)

    train(
        args,
        dataset,
        models,
        optimisers,
        frame_samplers
    )

    # with open(args.results_pkl, 'wb') as f:
    #     pickle.dump(results, f)
    

def test():
    return 0

def train(
    args,
    dataset: Dataset,
    models: List[nn.Module],
    optimisers: List[Adam],
    frame_samplers: List[RandomSampler],
):
    assert len(models) == len(optimisers)
    assert len(models) == len(frame_samplers)

    

    if args.val_features_pkl:
        trainloader = DataLoader(MultiPickleDataset(args.features_pkl), batch_size=args.batch_size, collate_fn=no_collate)
        testloader = DataLoader(MultiPickleDataset(args.val_features_pkl), batch_size=args.batch_size, collate_fn=no_collate)
    else:
        if args.train_test_split >= 0 and args.train_test_split <= 1:
            trainloader, testloader = train_test_loader(dataset, args.batch_size, args.train_test_split)
        else:
            raise ValueError(f"train / test split: {args.train_test_split} is an invalid percentage")

    writer = SummaryWriter(f'datasets/epic/runs/epic_mtrn_max-frames={args.max_frames}'f'_epochs={args.epoch}'f'_batch_size={args.batch_size}'f'_type={args.type}', flush_secs=1)

    training_result = []
    testing_result = []

    print(f"Training {args.type}s"f" for {args.min_frames}"f" - {args.max_frames}"f" frames...")

    for i in tqdm( # m, o, f
        # zip(models, optimisers, frame_samplers),
        range(args.min_frames-1, args.max_frames),
        unit=" model",
        dynamic_ncols=True
    ):
        classifier = EpicActionRecogintionShapleyClassifier(
            models[i],
            torch.device("cuda:0"),
            optimisers[i],
            frame_samplers[i],
            trainloader,
            testloader,
            args.type
        )

        # model_train_results = {
        #     'running_loss': [],
        #     'running_acc1': [],
        #     'running_acc5': [],
        #     'epoch_loss': [],
        #     'epoch_acc1': [],
        #     'epoch_acc5': []
        # }
        # model_test_results = {
        #     'running_loss': [],
        #     'running_acc1': [],
        #     'running_acc5': [],
        #     'epoch_loss': [],
        #     'epoch_acc1': [],
        #     'epoch_acc5': []
        # }

        liveloss = PlotLosses()

        for epoch in tqdm(
            range(args.epoch),
            unit=" epoch",
            dynamic_ncols=True
        ):
            logs = {}

            train_result = classifier.train_step()

            epoch_loss = sum(train_result[f'{models[i].frame_count}_loss']) / len(trainloader)
            epoch_acc1 = sum(train_result[f'{models[i].frame_count}_acc1']) / len(trainloader)
            epoch_acc5 = sum(train_result[f'{models[i].frame_count}_acc5']) / len(trainloader)

            # model_train_results['running_loss'].append(train_result[f'{models[i].frame_count}_loss'])
            # model_train_results['running_acc1'].append(train_result[f'{models[i].frame_count}_acc1'])
            # model_train_results['running_acc5'].append(train_result[f'{models[i].frame_count}_acc5'])
            # model_train_results['epoch_loss'].append(epoch_loss)
            # model_train_results['epoch_acc1'].append(epoch_acc1)
            # model_train_results['epoch_acc5'].append(epoch_acc5)

            writer.add_scalar(f'training loss frames={models[i].frame_count}', epoch_loss, epoch)
            writer.add_scalars('combined training loss', {f'loss frames={models[i].frame_count}': epoch_loss}, epoch)
            writer.add_scalars(f'training accuracy frames={models[i].frame_count}', {'acc1': epoch_acc1, 'acc5': epoch_acc5}, epoch)
            writer.add_scalars('combined training accuracy', {f'acc1 frames={models[i].frame_count}': epoch_acc1, f'acc5 frames={models[i].frame_count}': epoch_acc5}, epoch)

            test_result = classifier.test_step()

            epoch_loss_ = sum(test_result[f'{models[i].frame_count}_loss']) / len(testloader)
            epoch_acc1_ = sum(test_result[f'{models[i].frame_count}_acc1']) / len(testloader)
            epoch_acc5_ = sum(test_result[f'{models[i].frame_count}_acc5']) / len(testloader)

            # model_test_results['running_loss'].append(test_result[f'{models[i].frame_count}_loss'])
            # model_test_results['running_acc1'].append(test_result[f'{models[i].frame_count}_acc1'])
            # model_test_results['running_acc5'].append(test_result[f'{models[i].frame_count}_acc5'])
            # model_test_results['epoch_loss'].append(epoch_loss_)
            # model_test_results['epoch_acc1'].append(epoch_acc1_)
            # model_test_results['epoch_acc5'].append(epoch_acc5_)

            writer.add_scalar(f'testing loss frames={models[i].frame_count}', epoch_loss_, epoch)
            writer.add_scalars('combined testing loss', {f'loss frames={models[i].frame_count}': epoch_loss_}, epoch)
            writer.add_scalars(f'testing accuracy frames={models[i].frame_count}', {'acc1': epoch_acc1_, 'acc5': epoch_acc5_}, epoch)
            writer.add_scalars('combined testing accuracy', {f'acc1 frames={models[i].frame_count}': epoch_acc1_, f'acc5 frames={models[i].frame_count}': epoch_acc5_}, epoch)

            logs['loss'] = epoch_loss
            logs['accuracy'] = epoch_acc1
            logs['accuracy_5'] = epoch_acc5
            logs['val_loss'] = epoch_loss_
            logs['val_accuracy'] = epoch_acc1_
            logs['val_accuracy_5'] = epoch_acc5_

            # liveloss.update(logs)
            # liveloss.send()

        # training_result.append(model_train_results)
        # testing_result.append(model_test_results)

        classifier.save_parameters(args.model_params_dir / f'mtrn-frames={models[i].frame_count}'f'-type={args.type}.pt')
    
    # return {'training': training_result, 'testing': testing_result}

    # if args.save_params:
    #     classifier.save_parameters(args.save_params)

    # loss = np.concatenate(loss)

    # if args.save_fig:
    #     x = np.linspace(1, len(loss), len(loss), dtype=int)

    #     fig = go.Figure()

    #     fig.add_trace(go.Scatter(
    #         x=x,
    #         y=loss
    #     ))

    #     fig.update_layout(
    #         xaxis_title='batched steps',
    #         yaxis_title='loss',
    #         title='training performance'
    #     )
    #     fig.update_yaxes(type='log')
    #     fig.write_image(args.save_fig)


if __name__ == "__main__":
    main(parser.parse_args())

