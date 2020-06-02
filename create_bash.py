import os
from pathlib import Path


mdcg = ['NDCG@1', 'NDCG@2', 'NDCG@3', 'NDCG@4', 'NDCG@5',
        'NDCG@6', 'NDCG@7', 'NDCG@8', 'NDCG@9', 'NDCG@10']
p = ['P@1', 'P@2', 'P@3', 'P@4', 'P@5',
     'P@6', 'P@7', 'P@8', 'P@9', 'P@10', 'MAP']
index = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]
datasets = ['MQ2007semi', 'MQ2008semi']
# datasets = ['MQ2007']

algorithms = [(0, 'Mart'), (6, 'LambdaMart')]

with open('ir.bash', 'w') as f:
    f.write('set -x\n')

    for fold in index:
        for rid, ranker in algorithms:
            for dataset in datasets:
                for metric in p+mdcg:
                    path = Path(
                        f'output/{dataset}.{fold}.{ranker}.{metric}.output/')
                    path.mkdir(parents=True, exist_ok=True)
                    f.write(f'java -jar RankLib-2.9.jar -train {dataset}/{fold}/train_labeled.txt -test {dataset}/{fold}/test.txt -validate {dataset}/{fold}/vali.txt -ranker {rid} -metric2t {metric} -metric2T {metric} -save model/{dataset}.{fold}.{ranker}.{metric}.Baseline.txt > semi_performance_results/{dataset}.{fold}.{ranker}.{metric}.PerformanceResults.txt\n')
                    # f.write(f'java -jar RankLib-2.9.jar -train {dataset}/{fold}/train_with_newfeatures.txt -test {dataset}/{fold}/test_with_newfeatures.txt -validate {dataset}/{fold}/vali_with_newfeatures.txt -ranker {rid} -metric2t {metric} -metric2T {metric} -save model/{dataset}.{fold}.{ranker}.{metric}.txt > performance_results/{dataset}.{fold}.{ranker}.{metric}.AddNewFeatures.PerformanceResults.txt\n')
                    f.write(f'java -jar RankLib-2.9.jar -train {dataset}/{fold}/train_predict.txt -test {dataset}/{fold}/test.txt -validate {dataset}/{fold}/vali.txt -ranker {rid} -metric2t {metric} -metric2T {metric} -save model/{dataset}.{fold}.{ranker}.{metric}.txt > semi_performance_results/{dataset}.{fold}.{ranker}.{metric}.Predict.PerformanceResults.txt\n')

                    f.write(
                        f'java -jar RankLib-2.9.jar -load model/{dataset}.{fold}.{ranker}.{metric}.Baseline.txt -test {dataset}/{fold}/test.txt -metric2T {metric} -idv output/{dataset}.{fold}.{ranker}.{metric}.output/{dataset}.{fold}.{ranker}.{metric}.Baseline.txt\n')
                    # f.write(
                    #     f'java -jar RankLib-2.9.jar -load model/{dataset}.{fold}.{ranker}.{metric}.txt -test {dataset}/{fold}/test_with_newfeatures.txt -metric2T {metric} -idv output/{dataset}.{fold}.{ranker}.{metric}.output/{dataset}.{fold}.{ranker}.{metric}.txt\n')
                    f.write(
                        f'java -jar RankLib-2.9.jar -load model/{dataset}.{fold}.{ranker}.{metric}.txt -test {dataset}/{fold}/test.txt -metric2T {metric} -idv output/{dataset}.{fold}.{ranker}.{metric}.output/{dataset}.{fold}.{ranker}.{metric}.txt\n')

                    f.write(
                        f'java -cp RankLib-2.9.jar ciir.umass.edu.eval.Analyzer -all output/{dataset}.{fold}.{ranker}.{metric}.output/ -base {dataset}.{fold}.{ranker}.{metric}.Baseline.txt > analysis/{dataset}.{fold}.{ranker}.{metric}.Analysis.txt\n')
