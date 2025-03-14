import argparse
import importlib
import random
import sys,os,json

import mlflow

def setup():
    # Ask the user for the path to their code
    code_path = input("Please enter the full path to your code directory: ")
    save_path = input("Please enter the full path where you need assets to be saved: ")

    # Validate the path
    if not os.path.isdir(code_path):
        print("The specified path is not a valid directory.")
        return

    # Add the path to sys.path
    sys.path.append(code_path)

    # Save the path to a configuration file
    config = {'CODE_PATH': f'{code_path}','SAVE_PATH': f'{save_path}'}
    with open('config.json', 'w') as f:
        json.dump(config, f)

    print(f"Path '{code_path}' has been added to sys.path and saved to config.json")
    
def parse_args(args=None):
    
    parser = argparse.ArgumentParser(description="runs a multimodal fusion model ")
    parser.add_argument("--model_type", type=str, choices=['maxcorrmgnn','multiplex','rGCN', 'mGNN', 'multidimGCN', 'multibehGNN', 'GCN', 'latent_graph', 'nofusion', 'simple_fusion', 'metric_fusion', 'transformer'], 
                        help="specify one of the fusion models: 'maxcorrmgnn','multiplex','rGCN', 'mGNN', 'multidimgcn', 'multibehGNN', 'GCN', 'latent_graph', 'nofusion', 'simple_fusion', 'metric_fusion', 'transformer' ")
   
    parser.add_argument("--seed", type=int, choices=range(11), 
                        help="Seed value (0-10) for generating CV split")


    parser.add_argument("--predictor_type", type=str, choices=['GNN','MLP','multiplex-like-graphs', 'reduced-modality-hetero-graphs'],
                        help="required argument for maxcorrmgnn and rGCN baseline, specifies type of predictor, either GNN or MLP for maxcorrmgnn variants, and 'multiplex-like-graphs' or 'reduced-modality-hetero-graphs' for rGCN, will be ignored for other baselines")
    
    parser.add_argument("--loss_tradeoff", type=str, 
                        help="loss tradeoff, set between 0-1 for maxcorrmgnn, setting this to 0.0 performs sequential training of HGR-projectors and MultiGNN")
    
    parser.add_argument("--use_mlflow_tracking", type=int, choices = [0,1], 
                        help="Use MLflow or not")
    
    parser.add_argument("--process_graphs", type=int, choices = [0,1], 
                        help="Create graph objects from raw data using autoencoder processing")
   
    return parser.parse_args(args)
       
def main(args):
    
    if float(args.loss_tradeoff)>1.0:
        print(f"Invalid Spefication of tradeoff value,select between 0-1 range")
        sys.exit(1)

    if args.process_graphs==1:
        
        script_name = f"Run_graph_prep"
        script = importlib.import_module(script_name)
        
        script.main(args.seed)
    
    if args.use_mlflow_tracking==1:
        import mlflow
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8081")
    else:
        import mlflow
        mlflow.set_tracking_uri(uri="")
        mlflow.autolog(disable=True)
    
    #run predictors
    try:
        script_name = f"Run_{args.model_type}"
        script = importlib.import_module(script_name)
        
        script.main(args.seed, args.predictor_type,round(float(args.loss_tradeoff),2))
        
    except ImportError:
        print(f"Error: Script Run_{model_type} not found.")
        sys.exit(1)

if __name__ == "__main__":
    
    args = parse_args()
    setup()
 
    main(args)