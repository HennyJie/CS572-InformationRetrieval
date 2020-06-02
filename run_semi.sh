set -o xtrace
# for i in 1 2 3 4 5
# do
# for j in 1 2 3 4 5 6 7 8 9 10
# do
# java -jar RankLib-2.9.jar -train MQ2007semi/Fold$i/train_labeled.txt -test MQ2007semi/Fold$i/test.txt -validate MQ2007semi/Fold$i/vali.txt -ranker 6 -metric2t NDCG@$j -metric2T NDCG@$j -save semimodel/MQ2007semi.Fold$i.LambdaMart.NDCG@$j.Baseline.txt > semi_performance_results/MQ2007semi.Fold$i.LambdaMart.NDCG@$j.PerformanceResults.txt
# done
# done

for i in 1 
do
for j in 1 
do
java -jar RankLib-2.9.jar -load semimodel/MQ2007semi.Fold$i.LambdaMart.NDCG@$j.Baseline.txt -rank rank.txt -score score.txt

# java -jar RankLib-2.9.jar -load model/MQ2007.Fold$i.LambdaMart.NDCG@$j.txt -test MQ2007/Fold$i/test_with_newfeatures.txt -metric2T NDCG@$j -idv output/MQ2007.Fold$i.LambdaMart.NDCG@$j.output/MQ2007.Fold$i.LambdaMart.NDCG@$j.txt

# java -cp RankLib-2.9.jar ciir.umass.edu.eval.Analyzer -all output/MQ2007.Fold$i.LambdaMart.NDCG@$j.output/ -base MQ2007.Fold$i.LambdaMart.NDCG@$j.Baseline.txt > analysis/MQ2007.Fold$i.LambdaMart.NDCG@$j.Analysis.txt
done
done
