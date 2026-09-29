import os
import tensorflow as tf
import keras
from keras import layers, models
import matplotlib.pyplot as plt

"""
SOB/adenosis/SOB_B_A_14-22549G/200X : 32
SOB/adenosis/SOB_B_A_14-22549G/400X : 29
SOB/adenosis/SOB_B_A_14-22549G/100X : 34
SOB/adenosis/SOB_B_A_14-22549G/40X : 35
SOB/adenosis/SOB_B_A_14-22549CD/200X : 31
SOB/adenosis/SOB_B_A_14-22549CD/400X : 30
SOB/adenosis/SOB_B_A_14-22549CD/100X : 36
SOB/adenosis/SOB_B_A_14-22549CD/40X : 35
SOB/adenosis/SOB_B_A_14-22549AB/200X : 32
SOB/adenosis/SOB_B_A_14-22549AB/400X : 30
SOB/adenosis/SOB_B_A_14-22549AB/100X : 30
SOB/adenosis/SOB_B_A_14-22549AB/40X : 29
SOB/adenosis/SOB_B_A_14-29960CD/200X : 16
SOB/adenosis/SOB_B_A_14-29960CD/400X : 17
SOB/adenosis/SOB_B_A_14-29960CD/100X : 13
SOB/adenosis/SOB_B_A_14-29960CD/40X : 15


SOB/fibroadenoma/SOB_B_F_14-14134/200X : 38
SOB/fibroadenoma/SOB_B_F_14-14134/400X : 37
SOB/fibroadenoma/SOB_B_F_14-14134/100X : 31
SOB/fibroadenoma/SOB_B_F_14-14134/40X : 36
SOB/fibroadenoma/SOB_B_F_14-14134E/200X : 21
SOB/fibroadenoma/SOB_B_F_14-14134E/400X : 12
SOB/fibroadenoma/SOB_B_F_14-14134E/100X : 32
SOB/fibroadenoma/SOB_B_F_14-14134E/40X : 29
SOB/fibroadenoma/SOB_B_F_14-25197/200X : 32
SOB/fibroadenoma/SOB_B_F_14-25197/400X : 33
SOB/fibroadenoma/SOB_B_F_14-25197/100X : 33
SOB/fibroadenoma/SOB_B_F_14-25197/40X : 22
SOB/fibroadenoma/SOB_B_F_14-29960AB/200X : 18
SOB/fibroadenoma/SOB_B_F_14-29960AB/400X : 17
SOB/fibroadenoma/SOB_B_F_14-29960AB/100X : 15
SOB/fibroadenoma/SOB_B_F_14-29960AB/40X : 18
SOB/fibroadenoma/SOB_B_F_14-23060CD/200X : 16
SOB/fibroadenoma/SOB_B_F_14-23060CD/400X : 16
SOB/fibroadenoma/SOB_B_F_14-23060CD/100X : 11
SOB/fibroadenoma/SOB_B_F_14-23060CD/40X : 14
SOB/fibroadenoma/SOB_B_F_14-23060AB/200X : 18
SOB/fibroadenoma/SOB_B_F_14-23060AB/400X : 18
SOB/fibroadenoma/SOB_B_F_14-23060AB/100X : 22
SOB/fibroadenoma/SOB_B_F_14-23060AB/40X : 23
SOB/fibroadenoma/SOB_B_F_14-9133/200X : 38
SOB/fibroadenoma/SOB_B_F_14-9133/400X : 26
SOB/fibroadenoma/SOB_B_F_14-9133/100X : 36
SOB/fibroadenoma/SOB_B_F_14-9133/40X : 26
SOB/fibroadenoma/SOB_B_F_14-23222AB/200X : 17
SOB/fibroadenoma/SOB_B_F_14-23222AB/400X : 18
SOB/fibroadenoma/SOB_B_F_14-23222AB/100X : 16
SOB/fibroadenoma/SOB_B_F_14-23222AB/40X : 17
SOB/fibroadenoma/SOB_B_F_14-21998CD/200X : 37
SOB/fibroadenoma/SOB_B_F_14-21998CD/400X : 32
SOB/fibroadenoma/SOB_B_F_14-21998CD/100X : 31
SOB/fibroadenoma/SOB_B_F_14-21998CD/40X : 37
SOB/fibroadenoma/SOB_B_F_14-21998EF/200X : 29
SOB/fibroadenoma/SOB_B_F_14-21998EF/400X : 28
SOB/fibroadenoma/SOB_B_F_14-21998EF/100X : 33
SOB/fibroadenoma/SOB_B_F_14-21998EF/40X : 31


SOB/phyllodes_tumor/SOB_B_PT_14-22704/200X : 42
SOB/phyllodes_tumor/SOB_B_PT_14-22704/400X : 39
SOB/phyllodes_tumor/SOB_B_PT_14-22704/100X : 39
SOB/phyllodes_tumor/SOB_B_PT_14-22704/40X : 38
SOB/phyllodes_tumor/SOB_B_PT_14-21998AB/200X : 52
SOB/phyllodes_tumor/SOB_B_PT_14-21998AB/400X : 59
SOB/phyllodes_tumor/SOB_B_PT_14-21998AB/100X : 66
SOB/phyllodes_tumor/SOB_B_PT_14-21998AB/40X : 58
SOB/phyllodes_tumor/SOB_B_PT_14-29315EF/200X : 14
SOB/phyllodes_tumor/SOB_B_PT_14-29315EF/400X : 17
SOB/phyllodes_tumor/SOB_B_PT_14-29315EF/100X : 16
SOB/phyllodes_tumor/SOB_B_PT_14-29315EF/40X : 13


SOB/tubular_adenoma/SOB_B_TA_14-21978AB/200X : 17
SOB/tubular_adenoma/SOB_B_TA_14-21978AB/400X : 14
SOB/tubular_adenoma/SOB_B_TA_14-21978AB/100X : 18
SOB/tubular_adenoma/SOB_B_TA_14-21978AB/40X : 16
SOB/tubular_adenoma/SOB_B_TA_14-3411F/200X : 16
SOB/tubular_adenoma/SOB_B_TA_14-3411F/400X : 17
SOB/tubular_adenoma/SOB_B_TA_14-3411F/100X : 17
SOB/tubular_adenoma/SOB_B_TA_14-3411F/40X : 16
SOB/tubular_adenoma/SOB_B_TA_14-16184/200X : 35
SOB/tubular_adenoma/SOB_B_TA_14-16184/400X : 24
SOB/tubular_adenoma/SOB_B_TA_14-16184/100X : 34
SOB/tubular_adenoma/SOB_B_TA_14-16184/40X : 39
SOB/tubular_adenoma/SOB_B_TA_14-15275/200X : 12
SOB/tubular_adenoma/SOB_B_TA_14-15275/400X : 15
SOB/tubular_adenoma/SOB_B_TA_14-15275/100X : 14
SOB/tubular_adenoma/SOB_B_TA_14-15275/40X : 12
SOB/tubular_adenoma/SOB_B_TA_14-16184CD/200X : 23
SOB/tubular_adenoma/SOB_B_TA_14-16184CD/400X : 29
SOB/tubular_adenoma/SOB_B_TA_14-16184CD/100X : 37
SOB/tubular_adenoma/SOB_B_TA_14-16184CD/40X : 35
SOB/tubular_adenoma/SOB_B_TA_14-19854C/200X : 16
SOB/tubular_adenoma/SOB_B_TA_14-19854C/400X : 16
SOB/tubular_adenoma/SOB_B_TA_14-19854C/100X : 16
SOB/tubular_adenoma/SOB_B_TA_14-19854C/40X : 16
SOB/tubular_adenoma/SOB_B_TA_14-13200/200X : 21
SOB/tubular_adenoma/SOB_B_TA_14-13200/400X : 15
SOB/tubular_adenoma/SOB_B_TA_14-13200/100X : 14
SOB/tubular_adenoma/SOB_B_TA_14-13200/40X : 15


SOB/ductal_carcinoma/SOB_M_DC_14-18650/200X : 28
SOB/ductal_carcinoma/SOB_M_DC_14-18650/400X : 29
SOB/ductal_carcinoma/SOB_M_DC_14-18650/100X : 22
SOB/ductal_carcinoma/SOB_M_DC_14-18650/40X : 29
SOB/ductal_carcinoma/SOB_M_DC_14-15572/200X : 19
SOB/ductal_carcinoma/SOB_M_DC_14-15572/400X : 15
SOB/ductal_carcinoma/SOB_M_DC_14-15572/100X : 23
SOB/ductal_carcinoma/SOB_M_DC_14-15572/40X : 23
SOB/ductal_carcinoma/SOB_M_DC_14-2980/200X : 21
SOB/ductal_carcinoma/SOB_M_DC_14-2980/400X : 18
SOB/ductal_carcinoma/SOB_M_DC_14-2980/100X : 25
SOB/ductal_carcinoma/SOB_M_DC_14-2980/40X : 20
SOB/ductal_carcinoma/SOB_M_DC_14-15792/200X : 12
SOB/ductal_carcinoma/SOB_M_DC_14-15792/400X : 15
SOB/ductal_carcinoma/SOB_M_DC_14-15792/100X : 14
SOB/ductal_carcinoma/SOB_M_DC_14-15792/40X : 14
SOB/ductal_carcinoma/SOB_M_DC_14-13412/200X : 32
SOB/ductal_carcinoma/SOB_M_DC_14-13412/400X : 26
SOB/ductal_carcinoma/SOB_M_DC_14-13412/100X : 33
SOB/ductal_carcinoma/SOB_M_DC_14-13412/40X : 32
SOB/ductal_carcinoma/SOB_M_DC_14-6241/200X : 12
SOB/ductal_carcinoma/SOB_M_DC_14-6241/400X : 13
SOB/ductal_carcinoma/SOB_M_DC_14-6241/100X : 21
SOB/ductal_carcinoma/SOB_M_DC_14-6241/40X : 32
SOB/ductal_carcinoma/SOB_M_DC_14-11520/200X : 24
SOB/ductal_carcinoma/SOB_M_DC_14-11520/400X : 23
SOB/ductal_carcinoma/SOB_M_DC_14-11520/100X : 26
SOB/ductal_carcinoma/SOB_M_DC_14-11520/40X : 25
SOB/ductal_carcinoma/SOB_M_DC_14-5694/200X : 21
SOB/ductal_carcinoma/SOB_M_DC_14-5694/400X : 26
SOB/ductal_carcinoma/SOB_M_DC_14-5694/100X : 21
SOB/ductal_carcinoma/SOB_M_DC_14-5694/40X : 22
SOB/ductal_carcinoma/SOB_M_DC_14-5695/200X : 22
SOB/ductal_carcinoma/SOB_M_DC_14-5695/400X : 24
SOB/ductal_carcinoma/SOB_M_DC_14-5695/100X : 21
SOB/ductal_carcinoma/SOB_M_DC_14-5695/40X : 23
SOB/ductal_carcinoma/SOB_M_DC_14-16716/200X : 34
SOB/ductal_carcinoma/SOB_M_DC_14-16716/400X : 29
SOB/ductal_carcinoma/SOB_M_DC_14-16716/100X : 36
SOB/ductal_carcinoma/SOB_M_DC_14-16716/40X : 38
SOB/ductal_carcinoma/SOB_M_DC_14-2773/200X : 35
SOB/ductal_carcinoma/SOB_M_DC_14-2773/400X : 38
SOB/ductal_carcinoma/SOB_M_DC_14-2773/100X : 41
SOB/ductal_carcinoma/SOB_M_DC_14-2773/40X : 30
SOB/ductal_carcinoma/SOB_M_DC_14-17614/200X : 33
SOB/ductal_carcinoma/SOB_M_DC_14-17614/400X : 28
SOB/ductal_carcinoma/SOB_M_DC_14-17614/100X : 29
SOB/ductal_carcinoma/SOB_M_DC_14-17614/40X : 39
SOB/ductal_carcinoma/SOB_M_DC_14-2523/200X : 18
SOB/ductal_carcinoma/SOB_M_DC_14-2523/400X : 24
SOB/ductal_carcinoma/SOB_M_DC_14-2523/100X : 23
SOB/ductal_carcinoma/SOB_M_DC_14-2523/40X : 10
SOB/ductal_carcinoma/SOB_M_DC_14-4364/200X : 25
SOB/ductal_carcinoma/SOB_M_DC_14-4364/400X : 8
SOB/ductal_carcinoma/SOB_M_DC_14-4364/100X : 10
SOB/ductal_carcinoma/SOB_M_DC_14-4364/40X : 1
SOB/ductal_carcinoma/SOB_M_DC_14-14015/200X : 30
SOB/ductal_carcinoma/SOB_M_DC_14-14015/400X : 20
SOB/ductal_carcinoma/SOB_M_DC_14-14015/100X : 27
SOB/ductal_carcinoma/SOB_M_DC_14-14015/40X : 18
SOB/ductal_carcinoma/SOB_M_DC_14-16336/200X : 19
SOB/ductal_carcinoma/SOB_M_DC_14-16336/400X : 9
SOB/ductal_carcinoma/SOB_M_DC_14-16336/100X : 19
SOB/ductal_carcinoma/SOB_M_DC_14-16336/40X : 13
SOB/ductal_carcinoma/SOB_M_DC_14-15696/200X : 22
SOB/ductal_carcinoma/SOB_M_DC_14-15696/400X : 15
SOB/ductal_carcinoma/SOB_M_DC_14-15696/100X : 23
SOB/ductal_carcinoma/SOB_M_DC_14-15696/40X : 23
SOB/ductal_carcinoma/SOB_M_DC_14-12312/200X : 35
SOB/ductal_carcinoma/SOB_M_DC_14-12312/400X : 38
SOB/ductal_carcinoma/SOB_M_DC_14-12312/100X : 41
SOB/ductal_carcinoma/SOB_M_DC_14-12312/40X : 32
SOB/ductal_carcinoma/SOB_M_DC_14-4372/200X : 17
SOB/ductal_carcinoma/SOB_M_DC_14-4372/400X : 24
SOB/ductal_carcinoma/SOB_M_DC_14-4372/100X : 22
SOB/ductal_carcinoma/SOB_M_DC_14-4372/40X : 21
SOB/ductal_carcinoma/SOB_M_DC_14-11031/200X : 14
SOB/ductal_carcinoma/SOB_M_DC_14-11031/400X : 15
SOB/ductal_carcinoma/SOB_M_DC_14-11031/100X : 17
SOB/ductal_carcinoma/SOB_M_DC_14-11031/40X : 14
SOB/ductal_carcinoma/SOB_M_DC_14-16188/200X : 21
SOB/ductal_carcinoma/SOB_M_DC_14-16188/400X : 23
SOB/ductal_carcinoma/SOB_M_DC_14-16188/100X : 19
SOB/ductal_carcinoma/SOB_M_DC_14-16188/40X : 24
SOB/ductal_carcinoma/SOB_M_DC_14-9461/200X : 62
SOB/ductal_carcinoma/SOB_M_DC_14-9461/400X : 31
SOB/ductal_carcinoma/SOB_M_DC_14-9461/100X : 36
SOB/ductal_carcinoma/SOB_M_DC_14-9461/40X : 26
SOB/ductal_carcinoma/SOB_M_DC_14-5287/200X : 21
SOB/ductal_carcinoma/SOB_M_DC_14-5287/400X : 23
SOB/ductal_carcinoma/SOB_M_DC_14-5287/100X : 21
SOB/ductal_carcinoma/SOB_M_DC_14-5287/40X : 23
SOB/ductal_carcinoma/SOB_M_DC_14-11951/200X : 28
SOB/ductal_carcinoma/SOB_M_DC_14-11951/400X : 20
SOB/ductal_carcinoma/SOB_M_DC_14-11951/100X : 25
SOB/ductal_carcinoma/SOB_M_DC_14-11951/40X : 27
SOB/ductal_carcinoma/SOB_M_DC_14-16601/200X : 10
SOB/ductal_carcinoma/SOB_M_DC_14-16601/400X : 13
SOB/ductal_carcinoma/SOB_M_DC_14-16601/100X : 10
SOB/ductal_carcinoma/SOB_M_DC_14-16601/40X : 12
SOB/ductal_carcinoma/SOB_M_DC_14-13993/200X : 34
SOB/ductal_carcinoma/SOB_M_DC_14-13993/400X : 32
SOB/ductal_carcinoma/SOB_M_DC_14-13993/100X : 42
SOB/ductal_carcinoma/SOB_M_DC_14-13993/40X : 37
SOB/ductal_carcinoma/SOB_M_DC_14-10926/200X : 9
SOB/ductal_carcinoma/SOB_M_DC_14-10926/400X : 9
SOB/ductal_carcinoma/SOB_M_DC_14-10926/100X : 10
SOB/ductal_carcinoma/SOB_M_DC_14-10926/40X : 11
SOB/ductal_carcinoma/SOB_M_DC_14-20629/200X : 32
SOB/ductal_carcinoma/SOB_M_DC_14-20629/400X : 24
SOB/ductal_carcinoma/SOB_M_DC_14-20629/100X : 33
SOB/ductal_carcinoma/SOB_M_DC_14-20629/40X : 36
SOB/ductal_carcinoma/SOB_M_DC_14-16448/200X : 13
SOB/ductal_carcinoma/SOB_M_DC_14-16448/400X : 15
SOB/ductal_carcinoma/SOB_M_DC_14-16448/100X : 14
SOB/ductal_carcinoma/SOB_M_DC_14-16448/40X : 11
SOB/ductal_carcinoma/SOB_M_DC_14-16875/200X : 17
SOB/ductal_carcinoma/SOB_M_DC_14-16875/400X : 8
SOB/ductal_carcinoma/SOB_M_DC_14-16875/100X : 22
SOB/ductal_carcinoma/SOB_M_DC_14-16875/40X : 14
SOB/ductal_carcinoma/SOB_M_DC_14-3909/200X : 29
SOB/ductal_carcinoma/SOB_M_DC_14-3909/400X : 14
SOB/ductal_carcinoma/SOB_M_DC_14-3909/100X : 26
SOB/ductal_carcinoma/SOB_M_DC_14-3909/40X : 26
SOB/ductal_carcinoma/SOB_M_DC_14-17915/200X : 21
SOB/ductal_carcinoma/SOB_M_DC_14-17915/400X : 16
SOB/ductal_carcinoma/SOB_M_DC_14-17915/100X : 16
SOB/ductal_carcinoma/SOB_M_DC_14-17915/40X : 26
SOB/ductal_carcinoma/SOB_M_DC_14-2985/200X : 14
SOB/ductal_carcinoma/SOB_M_DC_14-2985/400X : 14
SOB/ductal_carcinoma/SOB_M_DC_14-2985/100X : 11
SOB/ductal_carcinoma/SOB_M_DC_14-2985/40X : 20
SOB/ductal_carcinoma/SOB_M_DC_14-17901/200X : 24
SOB/ductal_carcinoma/SOB_M_DC_14-17901/400X : 28
SOB/ductal_carcinoma/SOB_M_DC_14-17901/100X : 26
SOB/ductal_carcinoma/SOB_M_DC_14-17901/40X : 26
SOB/ductal_carcinoma/SOB_M_DC_14-20636/200X : 27
SOB/ductal_carcinoma/SOB_M_DC_14-20636/400X : 22
SOB/ductal_carcinoma/SOB_M_DC_14-20636/100X : 37
SOB/ductal_carcinoma/SOB_M_DC_14-20636/40X : 25
SOB/ductal_carcinoma/SOB_M_DC_14-8168/200X : 9
SOB/ductal_carcinoma/SOB_M_DC_14-8168/400X : 9
SOB/ductal_carcinoma/SOB_M_DC_14-8168/100X : 10
SOB/ductal_carcinoma/SOB_M_DC_14-8168/40X : 10
SOB/ductal_carcinoma/SOB_M_DC_14-14946/200X : 34
SOB/ductal_carcinoma/SOB_M_DC_14-14946/400X : 34
SOB/ductal_carcinoma/SOB_M_DC_14-14946/100X : 32
SOB/ductal_carcinoma/SOB_M_DC_14-14946/40X : 31
SOB/ductal_carcinoma/SOB_M_DC_14-14926/200X : 18
SOB/ductal_carcinoma/SOB_M_DC_14-14926/400X : 16
SOB/ductal_carcinoma/SOB_M_DC_14-14926/100X : 19
SOB/ductal_carcinoma/SOB_M_DC_14-14926/40X : 20


SOB/lobular_carcinoma/SOB_M_LC_14-12204/200X : 28
SOB/lobular_carcinoma/SOB_M_LC_14-12204/400X : 24
SOB/lobular_carcinoma/SOB_M_LC_14-12204/100X : 31
SOB/lobular_carcinoma/SOB_M_LC_14-12204/40X : 20
SOB/lobular_carcinoma/SOB_M_LC_14-15570C/200X : 30
SOB/lobular_carcinoma/SOB_M_LC_14-15570C/400X : 29
SOB/lobular_carcinoma/SOB_M_LC_14-15570C/100X : 35
SOB/lobular_carcinoma/SOB_M_LC_14-15570C/40X : 31
SOB/lobular_carcinoma/SOB_M_LC_14-15570/200X : 55
SOB/lobular_carcinoma/SOB_M_LC_14-15570/400X : 41
SOB/lobular_carcinoma/SOB_M_LC_14-15570/100X : 54
SOB/lobular_carcinoma/SOB_M_LC_14-15570/40X : 51
SOB/lobular_carcinoma/SOB_M_LC_14-16196/200X : 18
SOB/lobular_carcinoma/SOB_M_LC_14-16196/400X : 17
SOB/lobular_carcinoma/SOB_M_LC_14-16196/100X : 17
SOB/lobular_carcinoma/SOB_M_LC_14-16196/40X : 22
SOB/lobular_carcinoma/SOB_M_LC_14-13412/200X : 32
SOB/lobular_carcinoma/SOB_M_LC_14-13412/400X : 26
SOB/lobular_carcinoma/SOB_M_LC_14-13412/100X : 33
SOB/lobular_carcinoma/SOB_M_LC_14-13412/40X : 32


SOB/mucinous_carcinoma/SOB_M_MC_14-18842/200X : 16
SOB/mucinous_carcinoma/SOB_M_MC_14-18842/400X : 9
SOB/mucinous_carcinoma/SOB_M_MC_14-18842/100X : 22
SOB/mucinous_carcinoma/SOB_M_MC_14-18842/40X : 15
SOB/mucinous_carcinoma/SOB_M_MC_14-13418DE/200X : 14
SOB/mucinous_carcinoma/SOB_M_MC_14-13418DE/400X : 11
SOB/mucinous_carcinoma/SOB_M_MC_14-13418DE/100X : 15
SOB/mucinous_carcinoma/SOB_M_MC_14-13418DE/40X : 15
SOB/mucinous_carcinoma/SOB_M_MC_14-18842D/200X : 16
SOB/mucinous_carcinoma/SOB_M_MC_14-18842D/400X : 16
SOB/mucinous_carcinoma/SOB_M_MC_14-18842D/100X : 16
SOB/mucinous_carcinoma/SOB_M_MC_14-18842D/40X : 16
SOB/mucinous_carcinoma/SOB_M_MC_14-13413/200X : 23
SOB/mucinous_carcinoma/SOB_M_MC_14-13413/400X : 26
SOB/mucinous_carcinoma/SOB_M_MC_14-13413/100X : 41
SOB/mucinous_carcinoma/SOB_M_MC_14-13413/40X : 29
SOB/mucinous_carcinoma/SOB_M_MC_14-16456/200X : 47
SOB/mucinous_carcinoma/SOB_M_MC_14-16456/400X : 35
SOB/mucinous_carcinoma/SOB_M_MC_14-16456/100X : 50
SOB/mucinous_carcinoma/SOB_M_MC_14-16456/40X : 46
SOB/mucinous_carcinoma/SOB_M_MC_14-12773/200X : 21
SOB/mucinous_carcinoma/SOB_M_MC_14-12773/400X : 18
SOB/mucinous_carcinoma/SOB_M_MC_14-12773/100X : 25
SOB/mucinous_carcinoma/SOB_M_MC_14-12773/40X : 26
SOB/mucinous_carcinoma/SOB_M_MC_14-10147/200X : 22
SOB/mucinous_carcinoma/SOB_M_MC_14-10147/400X : 15
SOB/mucinous_carcinoma/SOB_M_MC_14-10147/100X : 12
SOB/mucinous_carcinoma/SOB_M_MC_14-10147/40X : 15
SOB/mucinous_carcinoma/SOB_M_MC_14-19979/200X : 21
SOB/mucinous_carcinoma/SOB_M_MC_14-19979/400X : 24
SOB/mucinous_carcinoma/SOB_M_MC_14-19979/100X : 27
SOB/mucinous_carcinoma/SOB_M_MC_14-19979/40X : 26
SOB/mucinous_carcinoma/SOB_M_MC_14-19979C/200X : 16
SOB/mucinous_carcinoma/SOB_M_MC_14-19979C/400X : 15
SOB/mucinous_carcinoma/SOB_M_MC_14-19979C/100X : 14
SOB/mucinous_carcinoma/SOB_M_MC_14-19979C/40X : 17


SOB/papillary_carcinoma/SOB_M_PC_14-19440/200X : 33
SOB/papillary_carcinoma/SOB_M_PC_14-19440/400X : 36
SOB/papillary_carcinoma/SOB_M_PC_14-19440/100X : 38
SOB/papillary_carcinoma/SOB_M_PC_14-19440/40X : 35
SOB/papillary_carcinoma/SOB_M_PC_15-190EF/200X : 15
SOB/papillary_carcinoma/SOB_M_PC_15-190EF/400X : 15
SOB/papillary_carcinoma/SOB_M_PC_15-190EF/100X : 16
SOB/papillary_carcinoma/SOB_M_PC_15-190EF/40X : 19
SOB/papillary_carcinoma/SOB_M_PC_14-15687B/200X : 14
SOB/papillary_carcinoma/SOB_M_PC_14-15687B/400X : 15
SOB/papillary_carcinoma/SOB_M_PC_14-15687B/100X : 16
SOB/papillary_carcinoma/SOB_M_PC_14-15687B/40X : 17
SOB/papillary_carcinoma/SOB_M_PC_14-12465/200X : 19
SOB/papillary_carcinoma/SOB_M_PC_14-12465/400X : 13
SOB/papillary_carcinoma/SOB_M_PC_14-12465/100X : 21
SOB/papillary_carcinoma/SOB_M_PC_14-12465/40X : 21
SOB/papillary_carcinoma/SOB_M_PC_14-15704/200X : 33
SOB/papillary_carcinoma/SOB_M_PC_14-15704/400X : 35
SOB/papillary_carcinoma/SOB_M_PC_14-15704/100X : 29
SOB/papillary_carcinoma/SOB_M_PC_14-15704/40X : 30
SOB/papillary_carcinoma/SOB_M_PC_14-9146/200X : 21
SOB/papillary_carcinoma/SOB_M_PC_14-9146/400X : 24
SOB/papillary_carcinoma/SOB_M_PC_14-9146/100X : 22
SOB/papillary_carcinoma/SOB_M_PC_14-9146/40X : 23
"""

DATA_DIR = "BreaKHis_v1/histology_slides/breast" 
BATCH_SIZE = 32
IMG_HEIGHT = 460
IMG_WIDTH = 700



train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Extract class names automatically based on your folder structure
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Loaded classes: {class_names}")

# Optimize data loading performance using caching and prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. BUILD THE MACHINE LEARNING MODEL
# ==========================================
model = models.Sequential([
    # Rescale pixel values from [0, 255] to [0, 1]
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    # Convolutional layers to look for patterns
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Flatten the 2D matrices into a 1D vector
    layers.Flatten(),
    
    # Dense layers for final classification
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# ==========================================
# 4. COMPILE AND TRAIN
# ==========================================
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# ==========================================
# 5. SAVE THE MODEL
# ==========================================
model.save('png_image_classifier.h5')
print("Model saved successfully!")