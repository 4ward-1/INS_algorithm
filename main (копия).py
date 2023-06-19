import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from   sympy.matrices import Matrix
import math
from scipy import interpolate

def load_data(id = 0): # Загрузка датасета показаний датчиков
    # X(E) - поперечная ось (вправо экрана) # Y(N) - продольная ось (вверх экрана) # Z(h) - вертикальная ось (перпендикулярно вперед к экрану)
    accelerometer = pd.read_csv('%s_TotalAcceleration.csv' % id) # при ускорении в сторону оси - положительное значение
    gyroscope     = pd.read_csv('%s_Gyroscope.csv' % id) # при вращении против часовой стрелки (если смотреть с конца оси) то положительное значение
    magnetometer  = pd.read_csv('%s_Magnetometer.csv' % id)
    GPS = pd.read_csv('%s_Location.csv' % id)
    return accelerometer, gyroscope, magnetometer, GPS

def constants():
    #const = {'g': 9.82, 'Rz': 6378000, 'T_GV': 45, 'T_K': 6, 'Kz': 0.8}
    const = {'g': 9.82, 'Rz': 6378000, 'T_GV': 12, 'T_K': 200, 'Kz': 10}
    return const

def razb(accelerometer, gyroscope, magnetometer, GPS):
    time_acc  = accelerometer['seconds_elapsed']
    acc_x = accelerometer['x']
    acc_y = accelerometer['y']
    acc_z = accelerometer['z']
    time_gyro = gyroscope['seconds_elapsed']
    gyro_x = gyroscope['x'] - np.mean(gyroscope['x'])
    gyro_y = gyroscope['y'] - np.mean(gyroscope['y'])
    gyro_z = gyroscope['z'] - np.mean(gyroscope['z'])
    time_magn = magnetometer['seconds_elapsed']
    magn_x = magnetometer['x']
    magn_y = magnetometer['y']
    magn_z = magnetometer['z']
    time_GPS = GPS['seconds_elapsed']
    B = GPS['latitude']
    L = GPS['longitude']
    return time_acc, acc_x, acc_y, acc_z, time_gyro, gyro_x, gyro_y, gyro_z, time_magn, magn_x, magn_y, magn_z, time_GPS, B, L

def interpolation(time_acc, acc_x, acc_y, acc_z, time_gyro, gyro_x, gyro_y, gyro_z, time_magn, magn_x, magn_y, magn_z, time_GPS, B, L):
    # Определение периода дискретизации и длительности датасета
    DT = round(((max(time_acc) / len(time_acc)) + (max(time_gyro) / len(time_gyro)) + (max(time_magn) / len(time_magn))) / 3, 3)
    mintime = max(min(time_acc), min(time_gyro), min(time_magn))
    maxtime = min(max(time_acc), max(time_gyro), max(time_magn))
    time = np.linspace(mintime, maxtime, num = int(maxtime / DT))
    acc_x_i  = interpolate.interp1d(time_acc,   acc_x, axis = 0, fill_value = "extrapolate")
    acc_y_i  = interpolate.interp1d(time_acc,   acc_y, axis = 0, fill_value = "extrapolate")
    acc_z_i  = interpolate.interp1d(time_acc,   acc_z, axis = 0, fill_value = "extrapolate")
    gyro_x_i = interpolate.interp1d(time_gyro, gyro_x, axis = 0, fill_value = "extrapolate")
    gyro_y_i = interpolate.interp1d(time_gyro, gyro_y, axis = 0, fill_value = "extrapolate")
    gyro_z_i = interpolate.interp1d(time_gyro, gyro_z, axis = 0, fill_value = "extrapolate")
    magn_x_i = interpolate.interp1d(time_magn, magn_x, axis = 0, fill_value = "extrapolate")
    magn_y_i = interpolate.interp1d(time_magn, magn_y, axis = 0, fill_value = "extrapolate")
    magn_z_i = interpolate.interp1d(time_magn, magn_z, axis = 0, fill_value = "extrapolate")
    B_i = interpolate.interp1d(time_GPS, B, axis = 0,  fill_value = "extrapolate")
    L_i = interpolate.interp1d(time_GPS, L, axis = 0,  fill_value = "extrapolate")

    acc_x_i  = acc_x_i(time)
    acc_y_i  = acc_y_i(time)
    acc_z_i  = acc_z_i(time)
    gyro_x_i = gyro_x_i(time)
    gyro_y_i = gyro_y_i(time)
    gyro_z_i = gyro_z_i(time)
    magn_x_i = magn_x_i(time)
    magn_y_i = magn_y_i(time)
    magn_z_i = magn_z_i(time)
    B_i = B_i(time)
    L_i = L_i(time)
    return DT, time, acc_x_i, acc_y_i, acc_z_i, gyro_x_i, gyro_y_i, gyro_z_i, magn_x_i, magn_y_i, magn_z_i, B_i, L_i

def initial_cond(DT, acc_x_i, acc_y_i, acc_z_i, B, L):
    V_Ei = 0; V_Ni = 0; V_hi = 0
    dS_Ei = 0; dS_Ni = 0; dS_hi = 0
    B0 = B[0]
    L0 = L[0]
    theta0 = math.asin(np.mean(acc_y_i[:int(1 / DT)]) / 9.82)
    gam0 = -math.atan2(np.mean(acc_x_i[:int(1 / DT)]), np.mean(acc_z_i[:int(1 / DT)]))
    return V_Ei, V_Ni, V_hi, dS_Ei, dS_Ni, dS_hi, B0, L0, theta0, gam0

def initial_Kurs(DT, theta0, gam0, B0, magn_x_i, magn_y_i, magn_z_i):
    # K0 из точного знания напряженности магнитного поля Земли
    m_N = 30 * math.cosh(B0 * math.pi)
    m_h = -60 * math.sin(B0 * math.pi)
    K0 = math.acos((np.mean(magn_y_i[:int(1 / DT)]) - m_h * math.sin(theta0)) / (m_N * math.cos(theta0)))

    if K0 < 0:
        K0 = 2*math.pi + K0  # Курс

    # K0 из проекцией вектора напряженнойсти на плоскость горизонта с упрощенной матрицей C1 без рыскания:
    A2 = Matrix([[1, 0, 0], [0, sym.cos('theta'), sym.sin('theta')], [0, sym.sin('-theta'), sym.cos('theta')]])  # Дифферент
    A3 = Matrix([[sym.cos('gam'), 0, sym.sin('-gam')], [0, 1, 0], [sym.sin('gam'), 0, sym.cos('gam')]])  # Крен
    A1 = A3 * A2
    C1 = Matrix.transpose(A1)

    C10_val = C1.subs({'theta': theta0, 'gam': gam0})
    c1_11 = C10_val[0, 0]; c1_12 = C10_val[0, 1]; c1_13 = C10_val[0, 2]
    c1_21 = C10_val[1, 0]; c1_22 = C10_val[1, 1]; c1_23 = C10_val[1, 2]
    magn_E1i = c1_11 * np.mean(magn_x_i[:int(1 / DT)]) + c1_12 * np.mean(magn_y_i[:int(1 / DT)]) + c1_13 * np.mean(magn_z_i[:int(1 / DT)])
    magn_N1i = c1_21 * np.mean(magn_x_i[:int(1 / DT)]) + c1_22 * np.mean(magn_y_i[:int(1 / DT)]) + c1_23 * np.mean(magn_z_i[:int(1 / DT)])
    K0 = -math.atan2(magn_E1i, magn_N1i)

    if K0 < 0:
        K0 = 2*math.pi + K0  # Курс
    return K0

def matrix_form():
    A1 = Matrix([[sym.cos('K'), sym.sin('-K'), 0], [sym.sin('K'), sym.cos('K'), 0], [0, 0, 1]])  # Рыскание
    A2 = Matrix([[1, 0, 0], [0, sym.cos('theta'), sym.sin('theta')], [0, sym.sin('-theta'), sym.cos('theta')]])  # Дифферент
    A3 = Matrix([[sym.cos('gam'), 0, sym.sin('-gam')], [0, 1, 0], [sym.sin('gam'), 0, sym.cos('gam')]])  # Крен
    A = A3 * A2 * A1  # Матрица перехода от ENh к XYZ
    C = Matrix.transpose(A)  # Матрица перехода от XYZ к ENh
    print(C)
    G = Matrix([[0], [0], ['g']])  # Вектор ускорения силы тяжести в ENh
    OmZ = Matrix([[0], ['OmE_N'], ['OmE_h']])  # Вектор угловой скорости вращения Земли в ENh
    Mgn = Matrix([[0], ['m_N'], ['m_h']])  # Вектор магнитного поля Земли в ENh
    G_xyz = A * G  # Вектор ускорения силы тяжести в XYZ
    OmE_xyz = A * OmZ  # Вектор угловой скорости вращения Земли в XYZ
    Mgn_xyz = A * Mgn  # Вектор магнитной индукции Земли в XYZ
    CC = Matrix([['c11', 'c12', 'c13'], ['c21', 'c22', 'c23'], ['c31', 'c32', 'c33']])  # Матрица направляющих косинусов
    Om_xyz_kos = Matrix([[0, '-om_z', 'om_y'], ['om_z', 0, '-om_x'], ['-om_y', 'om_x', 0]])  # Кососимметрическая матрица измеряемых угловых скоростей
    dC = CC * Om_xyz_kos  # Производные элементов матрицы направляющих косинусов
    return A, C, G, OmZ, Mgn, G_xyz, Mgn_xyz, CC, dC

if __name__ == '__main__':
    # 0. Выбор алгоритма обработки
    flag_alg = input('Выбереите алгоритм обработки исходных данных:\n 1 - Гиромагнитная курсовертикаль (уравнения Пуассона) \n'
                         ' 2 - Гиромагнитная курсовертикаль (кватернионы) \n 3 - Навигация без коррекции по GPS (уравнения Пуассона) \n'
                         ' 4 - Навигация без коррекции по GPS (кватернионы) \n 5 - Навигация с коррекцией по GPS (уравнения Пуассона) \n'
                         ' 6 - Навигация с коррекцией по GPS (кватернионы) \n')
    flag_acc = 0
    if flag_alg == '3' or flag_alg == '4' or flag_alg == '5' or flag_alg == '6':
        flag_acc = int(input('Убрать средние значения акселерометров в ENh?:\ 0 - нет, 1 - да\n'))

    flag_Kurs = int(input('Какой курс использовать при расчете линейных ускорений в ENh?: \n 1 - Гироскопический \n'
                      ' 2 - Магнитный \n'))

    data_id = input('Введите идентификатор данных: \n')
    # 1. Загрузка показаний датчиков и их систематических ошибок, а также констант
    [accelerometer, gyroscope, magnetometer, GPS] = load_data(data_id)
    const = constants()

    # 2. Разбиение данных датчиков и устранение их систематических ошибок
    [time_acc, acc_x, acc_y, acc_z, time_gyro, gyro_x, gyro_y, gyro_z, time_magn, magn_x, magn_y, magn_z, time_GPS, B, L] = razb(accelerometer, gyroscope, magnetometer, GPS)
    B = list(B)
    L = list(L)
    # 3. Интерполяция данных датчиков и расчет параметров
    [DT, time, acc_x_i, acc_y_i, acc_z_i, gyro_x_i, gyro_y_i, gyro_z_i, magn_x_i, magn_y_i, magn_z_i, B_i, L_i] = interpolation(time_acc, acc_x, acc_y, acc_z, time_gyro, gyro_x, gyro_y, gyro_z, time_magn, magn_x, magn_y, magn_z, time_GPS, B, L)
    if flag_alg == '1' or flag_alg == '2':
        b1 = DT/((const['T_GV'] + 2 * DT) * const['g'] * (const['T_GV'] ** 2))
        b2 = 2/((const['T_GV'] + 2 * DT) * const['g'] * const['T_GV'])
        b3 = const['T_GV']/(const['T_GV'] + 2 * DT)
        b4 = 2 * DT * const['Kz']/(const['T_K'] ** 2)
        b5 = 2 * const['Kz']/(const['T_K'])

    # 4. Формирование матриц
    [A, C, G, OmZ, Mgn, G_xyz, Mgn_xyz, CC, dC] = matrix_form()

    # 5. Начальная выставка
    [V_Ei, V_Ni, V_hi, dS_Ei, dS_Ni, dS_hi, B0, L0, theta0, gam0] = initial_cond(DT, acc_x_i, acc_y_i, acc_z_i, B, L)

    # 6. Начальный курс
    K0 = initial_Kurs(DT, theta0, gam0, B0, magn_x_i, magn_y_i, magn_z_i)

    # 7. Формирование углов качки, ускорений, линейных скоростей и перемещений и напряженностей магнитного поля в ENh
    theta = []; gam = []; Kgyr = []; Kmagn = []
    acc_E = []; acc_N = []; acc_h = []
    V_E = []; V_N = []; V_h = []
    dS_E = []; dS_N = []; dS_h = []
    dB = []; dL = []
    magn_E = []

    theta.append(theta0 * 180 / math.pi)
    gam.append(gam0 * 180 / math.pi)
    Kgyr.append(K0 * 180 / math.pi)
    Kmagn.append(K0 * 180 / math.pi)
    V_E.append(V_Ei)
    V_N.append(V_Ni)
    V_h.append(V_hi)
    dS_E.append(dS_Ei)
    dS_N.append(dS_Ni)
    dS_h.append(dS_hi)
    dB.append(0)
    dL.append(0)

    # 8. Формирование начальной матрицы C
    C0_val = C.subs({'K': K0, 'theta': theta0, 'gam': gam0})
    c11 = C0_val[0, 0]; c12 = C0_val[0, 1]; c13 = C0_val[0, 2]
    c21 = C0_val[1, 0]; c22 = C0_val[1, 1]; c23 = C0_val[1, 2]
    c31 = C0_val[2, 0]; c32 = C0_val[2, 1]; c33 = C0_val[2, 2]
    acc_E.append(c11 * acc_x_i[0] + c12 * acc_y_i[0] + c13 * acc_z_i[0])
    acc_N.append(c21 * acc_x_i[0] + c22 * acc_y_i[0] + c23 * acc_z_i[0])
    acc_h.append(c31 * acc_x_i[0] + c32 * acc_y_i[0] + c33 * acc_z_i[0] - 9.92)
    magn_E.append(c11 * magn_x_i[0] + c12 * magn_y_i[0] + c13 * magn_z_i[0])

    # 8. Средние значение ускорений в ENh
    mean_acc_E = 0; mean_acc_N = 0; mean_acc_h = 0

    # 9. Начальные условия сигналов коррекции
    if flag_alg == '1' or flag_alg == '2':
        gyro_N_corr1 = 0; gyro_E_corr1 = 0
        gyro_N_corr  = 0; gyro_E_corr  = 0; gyro_h_corr  = 0
        gyro_x_corr  = 0; gyro_y_corr  = 0; gyro_z_corr  = 0
        acc_Ei_p  = c11 * acc_x_i[0] + c12 * acc_y_i[0] + c13 * acc_z_i[0]
        acc_Ni_p  = c21 * acc_x_i[0] + c22 * acc_y_i[0] + c23 * acc_z_i[0]
        magn_Ei_p = c11 * magn_x_i[0] + c12 * magn_y_i[0] + c13 * magn_z_i[0]

    for i in range(1,len(time)):
        if flag_alg == '1' or flag_alg == '2' or flag_alg == '5' or flag_alg == '6':
            gyro_x_i[i] +=  gyro_x_corr
            gyro_y_i[i] -=  gyro_y_corr
            gyro_z_i[i] +=  gyro_z_corr

        dc11 =  c12 * gyro_z_i[i] - c13 * gyro_y_i[i]
        dc12 = -c11 * gyro_z_i[i] + c13 * gyro_x_i[i]
        dc13 =  c11 * gyro_y_i[i] - c12 * gyro_x_i[i]
        dc21 =  c22 * gyro_z_i[i] - c23 * gyro_y_i[i]
        dc22 = -c21 * gyro_z_i[i] + c23 * gyro_x_i[i]
        dc23 =  c21 * gyro_y_i[i] - c22 * gyro_x_i[i]
        dc31 =  c32 * gyro_z_i[i] - c33 * gyro_y_i[i]
        dc32 = -c31 * gyro_z_i[i] + c33 * gyro_x_i[i]
        dc33 =  c31 * gyro_y_i[i] - c32 * gyro_x_i[i]
        c11 += dc11 * DT; c12 += dc12 * DT; c13 += dc13 * DT
        c21 += dc21 * DT; c22 += dc22 * DT; c23 += dc23 * DT
        c31 += dc31 * DT; c32 += dc32 * DT; c33 += dc33 * DT

        magn_Ei = c11 * magn_x_i[i] + c12 * magn_y_i[i] + c13 * magn_z_i[i]
        magn_E.append(magn_Ei)

        theta_i = math.asin(c32)
        gam_i = -math.atan2(c31, c33)
        Kgyri = math.atan2(c12, c22)  # Курс гироскопический

        theta.append(theta_i * 180 / math.pi)
        gam.append(gam_i * 180 / math.pi)
        if Kgyri >= 0:
            Kgyr.append(Kgyri * 180 / math.pi)
        else:
            Kgyr.append((2 * math.pi + Kgyri) * 180 / math.pi)

        c1_11 = math.cos(gam_i); c1_12 = 0; c1_13 = math.sin(gam_i)
        c1_21 = math.sin(gam_i) * math.sin(theta_i); c1_22 = math.cos(theta_i); c1_23 = -math.sin(theta_i) * math.cos(gam_i)

        magn_E1i = c1_11 * magn_x_i[i] + c1_12 * magn_y_i[i] + c1_13 * magn_z_i[i]
        magn_N1i = c1_21 * magn_x_i[i] + c1_22 * magn_y_i[i] + c1_23 * magn_z_i[i]

        Kmagni = -math.atan2(magn_E1i, magn_N1i)  # Курс магнитный
        if Kmagni < 0:
            Kmagn.append((2 * math.pi + Kmagni) * 180 / math.pi)
        else:
            Kmagn.append(Kmagni * 180 / math.pi)

        if flag_Kurs == 1 or (flag_alg == '1' or flag_alg == '2'):
            acc_Ei = c11 * acc_x_i[i] + c12 * acc_y_i[i] + c13 * acc_z_i[i]
            acc_Ni = c21 * acc_x_i[i] + c22 * acc_y_i[i] + c23 * acc_z_i[i]
            acc_hi = c31 * acc_x_i[i] + c32 * acc_y_i[i] + c33 * acc_z_i[i] - const['g']
        else:
            # Формирование матрицы C с магнитным курсом вместо гироскопического
            c11_magn = math.sin(Kmagni) * math.sin(gam_i) * math.sin(theta_i) + math.cos(Kmagni) * math.cos(gam_i)
            c12_magn = math.sin(Kmagni) * math.cos(theta_i)
            c13_magn = -math.sin(Kmagni) * math.sin(theta_i) * math.cos(gam_i) + math.sin(gam_i) * math.cos(Kmagni)
            c21_magn = -math.sin(Kmagni) * math.cos(gam_i) + math.sin(gam_i) * math.sin(theta_i) * math.cos(Kmagni)
            c22_magn = math.cos(Kmagni) * math.cos(theta_i)
            c23_magn = -math.sin(Kmagni) * math.sin(gam_i) - math.sin(theta_i) * math.cos(Kmagni) * math.cos(gam_i)
            c31_magn = -math.sin(gam_i) * math.cos(theta_i)
            c32_magn = math.sin(theta_i)
            c33_magn = math.cos(gam_i) * math.cos(theta_i)

            acc_Ei = c11_magn * acc_x_i[i] + c12_magn * acc_y_i[i] + c13_magn * acc_z_i[i]
            acc_Ni = c21_magn * acc_x_i[i] + c22_magn * acc_y_i[i] + c23_magn * acc_z_i[i]
            acc_hi = c31_magn * acc_x_i[i] + c32_magn * acc_y_i[i] + c33_magn * acc_z_i[i] - const['g']

        if flag_acc == 1:
            mean_acc_E += acc_Ei; mean_acc_N += acc_Ni; mean_acc_h += acc_hi
            acc_Ei -= mean_acc_E / i # убираем постоянную составляющую восточного ускорения
            acc_Ni -= mean_acc_N / i # убираем постоянную составляющую северного ускорения
            acc_hi -= mean_acc_h / i # убираем постоянную составляющую вертикального ускорения

        acc_E.append(acc_Ei), acc_N.append(acc_Ni), acc_h.append(acc_hi)

        # Коррекция
        if flag_alg == '1' or flag_alg == '2':
            gyro_E_corr1 = gyro_E_corr1 * b3 + acc_Ni * b1 + (acc_Ni - acc_Ni_p) * b2
            gyro_N_corr1 = gyro_N_corr1 * b3 + acc_Ei * b1 + (acc_Ei - acc_Ei_p) * b2
            gyro_E_corr += DT * gyro_E_corr1
            gyro_N_corr += DT * gyro_N_corr1
            gyro_h_corr += magn_Ei * b4 + (magn_Ei - magn_Ei_p) * b5

            gyro_x_corr = c11 * gyro_E_corr + c21 * gyro_N_corr + c31 * gyro_h_corr
            gyro_y_corr = c12 * gyro_E_corr + c22 * gyro_N_corr + c32 * gyro_h_corr
            gyro_z_corr = c13 * gyro_E_corr + c23 * gyro_N_corr + c33 * gyro_h_corr

            acc_Ei_p  = acc_Ei
            acc_Ni_p  = acc_Ni
            magn_Ei_p = magn_Ei

        if (i*100/len(time)) % 5 and (i*100/len(time)) % 5 < 0.01:
            print(f'{int(i*100/len(time))} % ')

        if flag_alg == '3' or flag_alg == '4' or flag_alg == '5' or flag_alg == '6':
            V_Ei += acc_Ei * DT; V_Ni += acc_Ni * DT; V_hi += acc_hi * DT
            dS_Ei += V_Ei * DT; dS_Ni += V_Ni * DT; dS_hi += V_hi * DT

            dL.append(dS_Ei/(const['Rz']*math.cos(B0))); dB.append(dS_Ni/const['Rz'])
            V_E.append(V_Ei), V_N.append(V_Ni), V_h.append(V_hi)
            dS_E.append(dS_Ei), dS_N.append(dS_Ni), dS_h.append(dS_hi)

    if flag_alg == '1' or flag_alg == '2':
        plt.figure(1)
        plt.plot(time, theta, time, gam, time, Kgyr, time, Kmagn)
        plt.grid(color='red', linewidth=1, linestyle='--')
        plt.xlabel('t, с')
        plt.ylabel('град')
        plt.title('Дифферент, Крен, Курс (гироскопический и магнитный)')
        plt.figure(2)
        plt.plot(time, acc_E, time, acc_N, time, acc_h)
        plt.grid(color='red', linewidth=1, linestyle='--')
        plt.xlabel('t, с')
        plt.ylabel('м/с2')
        plt.title('Акселерометры в ГСТ')
        plt.figure(3)
        plt.plot(L, B, L0, B0, 'g*', L[-1], B[-1], 'r*')
        plt.grid(color='red', linewidth=1, linestyle='--')
        plt.xlabel('L, град')
        plt.ylabel('B, град')
        plt.title('Геодезические координаты по GPS')
        plt.show()

    if flag_alg == '3' or flag_alg == '4' or flag_alg == '5' or flag_alg == '6':
        plt.figure(1)
        plt.plot(time, theta, time, gam, time, Kgyr, time, Kmagn)
        plt.grid(color='red', linewidth=1, linestyle='--')
        plt.xlabel('t, с')
        plt.ylabel('град')
        plt.title('Дифферент, Крен, Курс (гироскопический и магнитный)')
        plt.figure(2)
        plt.plot(time, V_E, time, V_N, time, V_h)
        plt.grid(color='red', linewidth=1, linestyle='--')
        plt.xlabel('t, с')
        plt.ylabel('м/с')
        plt.title('Линейные скорости(ENh)')
        plt.figure(3)
        plt.plot(dS_E, dS_N)
        plt.grid(color='red', linewidth=1, linestyle='--')
        plt.xlabel('E, м')
        plt.ylabel('N, м')
        plt.title('Линейные перемещения(ENh)')
        plt.figure(4)
        plt.plot(time, acc_E, time, acc_N, time, acc_h)
        plt.grid(color='red', linewidth=1, linestyle='--')
        plt.xlabel('t, с')
        plt.ylabel('м/с2')
        plt.title('Акселерометры в ГСТ')
        plt.figure(5)
        L_inerc = [L0 + num * (180 / math.pi) for num in dL]
        B_inerc = [B0 + num * (180 / math.pi) for num in dB]
        plt.plot(L_inerc, B_inerc, L, B, L0, B0, 'g*', L[-1], B[-1], 'r*')
        plt.grid(color='red', linewidth=1, linestyle='--')
        plt.xlabel('L, град')
        plt.ylabel('B, град')
        plt.title('Геодезические координаты (инерциальные и GPS)')
        plt.show()