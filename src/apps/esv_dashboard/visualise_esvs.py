import argparse
import os

from pathlib import Path
import pandas as pd

import dash
from dash import Dash
from dash.exceptions import PreventUpdate
import flask

from apps.esv_dashboard.visualisation import Visualiser
from apps.esv_dashboard.result import Result, ShapleyValueResults

parser = argparse.ArgumentParser(
    description="Run web-based ESV visualisation tool",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("esvs_pkl", type=Path, help="Path to extracted ESVs")
parser.add_argument("dataset_root", type=Path, help="Path dataset folder of videos")
parser.add_argument(
    "verb_csv", type=Path, help="Path to verb CSV containing name,id entries"
)
parser.add_argument(
    "noun_csv", type=Path, help="Path to noun CSV containing name,id entries"
)
parser.add_argument(
    "verb_noun_link_dir", type=Path, help="Path to extracted verb noun links"
)
parser.add_argument(
    "--debug", default=True, type=bool, help="Enable Dash debug capabilities"
)
parser.add_argument(
    "--port", default=8050, type=int, help="Port for webserver to listen on"
)
parser.add_argument("--host", default="localhost", help="Host to bind to")

def main(args):

    args = parser.parse_args()
    dataset_dir: Path = args.dataset_root

    colours = {
        'rgb': {
            'yellow_20': 'rgba(244,160,0,0.1)',
            'blue_20': 'rgba(66,133,244,0.05)'
        },
        'hex': {
            'red': '#DB4437',
            'blue': '#4285F4',
            'yellow': '#F4B400',
            'green': '#0F9D58'
        }
    }

    verbs = pd.read_csv(args.verb_csv)
    nouns = pd.read_csv(args.noun_csv)

    verb2str = pd.Series(verbs.key.values,index=verbs.id).to_dict()
    noun2str = pd.Series(nouns.key.values,index=nouns.id).to_dict()

    verb_noun = pd.read_pickle(args.verb_noun_link_dir / 'verb_noun.pkl')
    verb_noun_classes = pd.read_pickle(args.verb_noun_link_dir / 'verb_noun_classes.pkl')
    verb_noun_narration = pd.read_pickle(args.verb_noun_link_dir /'verb_noun_classes_narration.pkl')

    results_dict = pd.read_pickle(args.esvs_pkl)

    title = "ESV Dashboard"

    results = ShapleyValueResults(results_dict)
    visualisation = Visualiser(
        results,
        colours, 
        verb2str, 
        noun2str, verb_noun,
        verb_noun_classes,
        verb_noun_narration, 
        dataset_dir, 
        title=title
    )

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = Dash(
        __name__,
        title="ESV Visualiser",
        update_title="Updating..." if args.debug else None,
        external_stylesheets=external_stylesheets,
    )

    visualisation.attach_to_app(app)
    app.run_server(host=args.host, debug=args.debug, port=args.port)

if __name__ == "__main__":
    main(parser.parse_args())

