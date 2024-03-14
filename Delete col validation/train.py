from delete_col_expriment import *
from model import *
import warnings
from param import *
def train(args):
    similarity_feature = loading_similarity_feature(args)
    edge_idx_dict, g = data_preprocess(args)

    print("*************************starting the train*****************************")
    print("***********************Iterate every 10 seconds *****************************")
    metric_result_list = []
    metric_result_list_str = []
    metric_result_list_str.append('AUC        AUPR       Acc       F1      pre        recall')
    for i in range(args.kfolds):
        model = Trifusion(args).to(args.device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        criterion = torch.nn.BCEWithLogitsLoss().to(args.device)

        print(f'###########################Fold {i + 1} of {args.kfolds}###########################')
        Record_res = []
        Record_res.append('AUC          AUPR         Acc         F1        pre          recall')
        model.train()
        for epoch in range(args.epoch):
            optimizer.zero_grad()

            out = model(args, similarity_feature, g, edge_idx_dict, edge_idx_dict[str(i)]['fold_train_edges_80p_80n'],
                        i).view(-1)

            loss = criterion(out, edge_idx_dict[str(i)]['fold_train_label_80p_80n'])
            loss.backward()
            optimizer.step()

            test_auc, metric_result, y_true, y_score = valid(args, model,similarity_feature,g,edge_idx_dict,edge_idx_dict[str(i)]['fold_valid_edges_20p_20n'],i
                                                                  )
            One_epoch_metric = '{:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f} '.format(*metric_result)
            Record_res.append(One_epoch_metric)
            if epoch + 1 == args.epoch:
                metric_result_list.append(metric_result)
                metric_result_list_str.append(One_epoch_metric)
            print('epoch {:03d} train_loss {:.8f} val_auc {:.4f} '.format(epoch + 1, loss.item(), test_auc))

    arr = np.array(metric_result_list)
    averages = np.round(np.mean(arr, axis=0), 4)
    metric_result_list_str.append('average:')
    metric_result_list_str.append('{:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f} '.format(*list(averages)))


    with open('result ' +'_'+ str(averages[0]) +'_.txt', 'w') as f:
        f.write('\n'.join(metric_result_list_str))
    return averages


def valid(args, model, similarity_feature, graph, edge_idx_dict, edge_label_index, i):
    lable = edge_idx_dict[str(i)]['fold_valid_label_20p_20n']

    model.eval()
    with torch.no_grad():
        out = model.encode(args, similarity_feature, graph, edge_idx_dict, i)
        res = model.decode(out, edge_label_index).view(-1).sigmoid()
        model.train()
    metric_result = caculate_metrics(lable.cpu().numpy(), res.cpu().numpy())
    my_acu = metrics.roc_auc_score(lable.cpu().numpy(), res.cpu().numpy())
    return my_acu, metric_result, lable, res

def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    average_result = train(args)
    print(average_result)
    print("finish")

if __name__ == '__main__':
    main()