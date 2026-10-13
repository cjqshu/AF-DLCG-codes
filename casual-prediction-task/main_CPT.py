# import sys
# sys.path.append("..")
import os
import pandas as pd
import numpy as np
import random
import torch
from sklearn import metrics
import configargparse
from torch.optim import lr_scheduler
import causal_discovery, data_loader, utils
from models.bnncc_regression import BnnccRegressor, bnncc_reg_loss, NncgcRegressor

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.options.display.float_format = '{:.3f}'.format


def get_parser():
    """get default arguments."""
    parser = configargparse.ArgumentParser(
        description="causal learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument('--dataset_name', default="ihdp", type=str, choices=["ihdp", "cpt", "jobs"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "cuda"])

    # network related
    parser.add_argument("--hidden_size", default=12, type=int)
    parser.add_argument("--hidden_layers", default=10, type=int)
    parser.add_argument("--output_size", default=1, type=int)
    
    # IRM related
    parser.add_argument("--use_irm", default=False, type=bool, help="use IRM model or not")
    parser.add_argument("--irm_penalty_weight", default=1.0, type=float, help="IRM penalty weight")
    parser.add_argument("--n_environments", default=2, type=int, help="number of environments for IRM")

    # data loading related
    parser.add_argument("--data_dir", default="data/", type=str, help="data directory")
    parser.add_argument("--rep_i", default=16, type=int, help="index number of the repetitive dataset is in [0, 999]")

    # training related
    parser.add_argument("--batch_size", default=10, type=int, help="batch size")
    parser.add_argument("--n_epoch", default=50, type=int, help="number of epochs")
    parser.add_argument("--early_stop", default=0, type=int, help="early stopping patience")

    # optimizer related
    parser.add_argument("--lr", default=1e-3, type=float)  # 1e-3 is equal to 1*10^(-3)=0.001
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)

    # experiments related
    parser.add_argument("--n_experiments", default=1, type=int, help="number of experiments")

    # save results
    parser.add_argument("--save_model", default=False, type=bool, help="save model or not")
    parser.add_argument("--save_dir", default="results_realworld/", type=str, help="directory of results")

    # baselines
    parser.add_argument("--baselines_status", default=False, type=bool, help="baselines or not")

    # ablations
    parser.add_argument("--ablations", default=False, type=bool, help="ablation or not")

    # predefined dag
    parser.add_argument("--predefined_dag", default=False, type=bool, help="predefined dag or not")

    return parser


def set_random_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(args, n_instances=170, n_vars=6, Y_noise=0.5, cv_splits=10):
    data = data_loader.DataLoader(
        data_path=args.data_dir,  # data_path="./data/"
        n_instances=n_instances,
        n_vars=n_vars,
        X_noise=0,
        Y_noise=Y_noise,
        test_perc=0.2,
        cv_splits=cv_splits,
        rep_i=args.rep_i,
    )

    data.load_dataset(args.dataset_name)

    return data


def load_fold_data(fold_dir, fold_num):
    train_path = os.path.join(fold_dir, f"fold_{fold_num}_train.csv")
    test_path = os.path.join(fold_dir, f"fold_{fold_num}_test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    class FoldData:
        def __init__(self, train_df, test_df):
            self.train_df = train_df
            self.test_df = test_df
            x_cols = [col for col in train_df.columns if col not in ['CPT_Y']]
            self.X_features = x_cols
            self.Y_feature = ['CPT_Y']
    
    data = FoldData(train_df, test_df)
    # return train_X, train_y, test_X, test_y
    return data
    
def data_preprocessing(train_df, test_df, X_features, y_feature):
    train_df = utils.normalize(train_df)
    test_df = utils.normalize(test_df)

    train_X = train_df[X_features].to_numpy()
    train_y = train_df[y_feature].to_numpy()
    test_X = test_df[X_features].to_numpy()
    test_y = test_df[y_feature].to_numpy()

    return train_X, train_y, test_X, test_y

def get_causal_groups(args, data):
    """Extract DAG"""
    dag_creator = causal_discovery.DAGCreator(data.train_df, data.X_features, data.Y_feature, method="icalingam",max_samples=1000, )

    if args.predefined_dag:
        dag_edges = dag_creator.return_predefined_dag("cpt")
    else:
        dag_edges = dag_creator.create_dag_edges()

    nodes_in_edges = [item for sublist in dag_edges for item in sublist]

    nodes_not_in_edges = [ x for x in dag_creator.X_features if x not in nodes_in_edges ]
    dag_edges.append(nodes_not_in_edges)

    results = dag_edges

    return results


def get_causal_groups_data(args, causal_groups, train_X, train_y, test_X, test_y, X_features):
    dag_edges = causal_groups
    dag_edges_idx = [[X_features.index(col) for col in sublist] for sublist in dag_edges]
    print(dag_edges_idx)

    structure_params = {}
    train_Xs = {}

    for i in range(len(dag_edges_idx)):
        structure_params[f"CG{i}"] = {
            "input_size": len(dag_edges_idx[i]),
            "hidden_size": args.hidden_size,
            "hidden_layers": args.hidden_layers,
            "output_size": args.output_size,
        }

        train_Xs[f"CG{i}"] = ( torch.from_numpy(utils.array_to_batch(train_X[:, dag_edges_idx[i]], args.batch_size))
            .to(torch.float32).to(args.device) )

    print(f"structure_params: {structure_params}")

    batched_train_y = ( torch.from_numpy(utils.array_to_batch(train_y, args.batch_size)).to(torch.float32).to(args.device) )

    test_Xs = {}
    for i in range(len(dag_edges_idx)):

        test_Xs[f"CG{i}"] = ( torch.from_numpy(utils.array_to_batch(test_X[:, dag_edges_idx[i]], len(test_X)))
            .to(torch.float32).to(args.device) )

        test_CG_shape = test_Xs[f"CG{i}"].shape

    batched_test_y = ( torch.from_numpy(utils.array_to_batch(test_y, len(test_y))).to(torch.float32).to(args.device) )

    return structure_params, train_Xs, batched_train_y, test_Xs, batched_test_y


def get_batch_ids(batched_train_y):
    train_batch_ids = np.random.choice(range(len(batched_train_y)), int(len(batched_train_y) * 1.0))

    return train_batch_ids


def get_model_name(args):
    if args.use_irm:
        return "IRMRegressor"
    elif args.ablations:
        return "BnnccnrRegressor"
    else:
        return "BnnccRegressor"


def get_model(args, structure_params):

    if args.use_irm:
        model = IRMRegressor(structure_params=structure_params)
        model_name = "IRMRegressor"
    elif args.ablations:
        model = NncgcRegressor(structure_params=structure_params)

        # model_name = "BnnccnrRegressor"
        model_name = "NncgcRegressor"
    else:
        model = BnnccRegressor(structure_params=structure_params)
        model_name = "BnnccRegressor"

    return model


def get_optimizer(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    return optimizer


def get_scheduler(optimizer):
    lambda1 = lambda x: 0.1 ** (x // 20)  
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, ])

    return scheduler


def train(i_epoch, n_epoch, model, train_batch_ids, train_Xs, batched_train_y, optimizer, 
          use_irm=False, env_data_list=None, irm_penalty_weight=1.0):
    model.train()

    losses = []
    for i in train_batch_ids:
        Xs_batch = {k: v[i] for k, v in train_Xs.items()}
        y_batch = batched_train_y[i]
        optimizer.zero_grad()
        output = model(Xs_batch)
        loss = bnncc_reg_loss(output, y_batch)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    train_losses = np.mean(losses)  # sum(losses) / len(losses)
    print(f"epoch: {i_epoch:>4}/{n_epoch} | train_losses: {train_losses:.3f}")

    return train_losses


def val(i_epoch, n_epoch, model, test_Xs, batched_test_y):  # validation
    model.eval()

    losses = []
    with torch.no_grad():
        test_Xs = {k: v[0] for k, v in test_Xs.items()}
        output = model(test_Xs)

        y_batch = batched_test_y[0]
        val_losses = bnncc_reg_loss(output, y_batch)

        y_true = y_batch.cpu().numpy() if torch.is_tensor(y_batch) else y_batch
        y_pred = output.cpu().numpy() if torch.is_tensor(output) else output
        
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(y_true, y_pred)
        val_r2 = metrics.r2_score(y_true, y_pred)
        print(f"epoch: {i_epoch:>4}/{n_epoch} |   val_losses: {val_losses:.3f} | rmse: {rmse:.3f} | mae: {mae:.3f} | r2: {val_r2:.3f}")

    return y_pred, val_losses, val_r2, rmse, mae, mse


def run_experiments(args, structure_params, train_batch_ids, train_Xs, batched_train_y, test_Xs, batched_test_y,
                    env_data_list=None):

    n_epoch = args.n_epoch
    n_experiments = args.n_experiments
    
    experiment_results = []
    
    for experiment in range(n_experiments):
        print(f"\nExperiment: {experiment + 1}")

        model = get_model(args, structure_params)
        model = model.to(args.device)
        optimizer = get_optimizer(args, model)

        best_val_r2 = -np.inf
        best_metrics = None
        
        for i_epoch in range(1, n_epoch + 1):
            # training process
            train_losses = train(i_epoch, n_epoch, model=model, train_batch_ids=train_batch_ids, train_Xs=train_Xs,
                  batched_train_y=batched_train_y, optimizer=optimizer,
                  use_irm=args.use_irm, env_data_list=env_data_list, 
                  irm_penalty_weight=args.irm_penalty_weight)
            # validating process
            pred_y, val_losses, val_r2, rmse, mae, mse = val(i_epoch, n_epoch, model=model, test_Xs=test_Xs, batched_test_y=batched_test_y)

            # learning rate scheduler
            scheduler = get_scheduler(optimizer)
            scheduler.step()
            
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_metrics = {
                    'train_loss': train_losses,
                    'val_loss': val_losses.item() if torch.is_tensor(val_losses) else val_losses,
                    'val_r2': val_r2,
                    'rmse': rmse,
                    'mae': mae,
                    'mse': mse,
                    'epoch': i_epoch
                }

        if best_metrics is not None:
            experiment_results.append(best_metrics)
            print(f"Experiment {experiment + 1} best results: R2={best_val_r2:.3f}, RMSE={best_metrics['rmse']:.3f}, MAE={best_metrics['mae']:.3f}")
        else:
            print(f"Experiment {experiment + 1} not found valid results")

    return experiment_results


def main():
    parser = get_parser()
    args = parser.parse_args()

    args.dataset_name = "cpt"
    print(f"starting {args.dataset_name}")

    fold_dir = "notebooks/CPT/fold/t7v3_seed-0-9"
    n_folds = 10

    args.predefined_dag = True
    args.ablations = True

    args.hidden_size = 6
    args.hidden_layers = 3
    args.batch_size = 2
    args.n_epoch = 300
    args.n_experiments = 10

    model_name = get_model_name(args)
    print(f"model name: {model_name}")
    
    all_results = {}
    for fold_num in range(1, n_folds + 1):
        print(f"\n{'='*60}")
        print(f"processing Fold {fold_num}/{n_folds}")
        print(f"{'='*60}\n")
        
        set_random_seed(seed=args.seed)
        
        try:
            data = load_fold_data(fold_dir, fold_num)
            print(f"download success Fold {fold_num} data")
            print(f"the size of training set: {len(data.train_df)}, the size of test set: {len(data.test_df)}")
        except Exception as e:
            print(f"download Fold {fold_num} data failure: {e}")
            continue
        
        train_df, test_df, X_features, y_feature = data.train_df, data.test_df, data.X_features, data.Y_feature
        train_X, train_y, test_X, test_y = data_preprocessing(train_df, test_df, X_features, y_feature)

        causal_groups = get_causal_groups(args, data)
        structure_params, train_Xs, batched_train_y, test_Xs, batched_test_y = get_causal_groups_data(
            args, causal_groups, train_X, train_y, test_X, test_y, X_features)

        train_batch_ids = get_batch_ids(batched_train_y)

        env_data_list = None

        fold_results = run_experiments(
            args, structure_params, train_batch_ids, train_Xs, batched_train_y, 
            test_Xs, batched_test_y, env_data_list=env_data_list
        )
        
        all_results[fold_num] = fold_results
        
        print(f"\nFold {fold_num} finished")
    
    print("\n" + "="*60)
    print("experiments summary")
    print("="*60)
    all_r2 = []
    all_rmse = []
    all_mae = []
    for fold_num, fold_results in all_results.items():
        if len(fold_results) > 0:
            r2_values = [r['val_r2'] for r in fold_results]
            rmse_values = [r['rmse'] for r in fold_results]
            mae_values = [r['mae'] for r in fold_results]
            all_r2.extend(r2_values)
            all_rmse.extend(rmse_values)
            all_mae.extend(mae_values)
            print(f"Fold {fold_num}: R2={np.mean(r2_values):.3f}±{np.std(r2_values):.3f}, "
                  f"RMSE={np.mean(rmse_values):.3f}±{np.std(rmse_values):.3f}, "
                  f"MAE={np.mean(mae_values):.3f}±{np.std(mae_values):.3f}")
    
    if len(all_r2) > 0:
        print(f"\ntotal: R2={np.mean(all_r2):.3f}±{np.std(all_r2):.3f}, "
              f"RMSE={np.mean(all_rmse):.3f}±{np.std(all_rmse):.3f}, "
              f"MAE={np.mean(all_mae):.3f}±{np.std(all_mae):.3f}")

if __name__ == "__main__":
    main()
