#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 @author: Chao
 @contact: zhaochaocs@gmail.com
 @time: 1/28/2019 1:30 PM
"""

import gurobipy


def ILPSolve(s, d, l, _lambda):
    input_len = len(s)

    MODEL = gurobipy.Model("Summary")

    # create variables
    x = MODEL.addVars(input_len, lb=0, ub=1, vtype=gurobipy.GRB.BINARY, name='x')
    y = MODEL.addVars(input_len, input_len, lb=0, ub=1, vtype=gurobipy.GRB.BINARY, name='y')

    # update the variable environment
    MODEL.update()

    # create the objective
    # MODEL.setObjective(x.prod(s) + 1 * y.prod(d), gurobipy.GRB.MAXIMIZE)
    if _lambda:
        MODEL.setObjective(x.prod(s) - \
                           _lambda * gurobipy.quicksum(
            y[i, j] * d[i][j] for i in range(input_len) for j in range(input_len)), \
                           gurobipy.GRB.MAXIMIZE)
    else:
        MODEL.setObjective(x.prod(s), gurobipy.GRB.MAXIMIZE)

    # create the constrains
    MODEL.addConstr(x.prod(l) <= 100)
    MODEL.addConstrs(y[i, j] >= x[i] + x[j] - 1 for i in range(input_len) for j in range(input_len))
    MODEL.addConstrs(y[i, j] <= (x[i] + x[j]) / 2.0 for i in range(input_len) for j in range(input_len))

    # run the model
    MODEL.optimize()

    x_val = []
    print("Obj:", MODEL.objVal)
    for v in MODEL.getVars():
        if 'x' in v.varName:
            x_val.append(v.x)
    return x_val
