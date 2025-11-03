import rpy2

import rpy2.robjects as robjects
robjects.r('install.packages(c("mlr3", "mlr3learners", "mlr3extralearners", "paradox", "checkmate", "lgr"))')

import rpy2.robjects as robjects

# Installer les packages (si ce n'est pas déjà fait)
"""robjects.r('''
    if (!require("mlr3")) install.packages("mlr3")
    if (!require("mlr3proba")) install.packages("mlr3proba")
    if (!require("mlr3learners")) install.packages("mlr3learners")
''')"""
"""
# Exécuter l'exemple
robjects.r('''
    library(mlr3)
    library(mlr3proba)
    library(mlr3learners)

    data("iris")
    task = as_task_classif(iris, target = "Species")
    learner = lrn("classif.ranger", predict_type = "prob")
    resampling = rsmp("cv", folds = 3)
    rr = resample(task, learner, resampling)
    print(rr$score())

    learner$train(task)
    prediction = learner$predict(task)
    head(prediction$score())
''')"""

