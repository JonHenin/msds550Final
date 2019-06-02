#### Results for controversial-comments

Best Model: LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)

Best Score: 0.956021052631579
Best Penalty: l2
Best C: 1.0
Best AUC: 0.5000342429477704

              precision    recall  f1-score   support

           0       0.96      1.00      0.98    226965
           1       0.14      0.00      0.00     10535

   micro avg       0.96      0.96      0.96    237500
   macro avg       0.55      0.50      0.49    237500
weighted avg       0.92      0.96      0.93    237500


Confusion Matrix:
[[226959      6]
 [ 10534      1]]
