import numpy as np

from regression_helpers import optimize_d_plus1, build_poly
from data_reader import load_clean_csv
from implementations import logistic_regression, least_squares

#y, tX, ids = load_clean_csv('data/clean_train_data.csv')
y, tX, ids = load_clean_csv('data/clean_short_train.csv')
#y_test, tX_test, ids_test = load_clean_csv('data/clean_short_test.csv')

tX = build_poly(tX, 1) / 1e6

#y = y/1e6

#d_best, p_best, min_te_loss, min_tr_loss, res_te, res_tr = optimize_d_plus1(y=y, x=tX, k_fold=10, method=logistic_regression, d_range=[1], param_range=[], *args)



w_best, loss_tr = logistic_regression(y, tX, 0.01, initial_w=False, max_iters=1000)
#w_best_2, loss_tr_2 = logistic_regression_SGD(y, tX, 0.8, initial_w=False, max_iters=500)
#w_best_3, loss_tr_3 = least_squares(y, tX)

#print(w_best_2, loss_tr_2)
#print(w_best_3, loss_tr_3)
print(w_best, loss_tr)