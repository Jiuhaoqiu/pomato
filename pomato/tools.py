import matplotlib.pyplot as plt
import pandas as pd

def find_xy_limits(list_plots):
    try:
        x_max = 0
        x_min = 0
        y_max = 0
        y_min = 0
        for plots in list_plots:
            tmpx = plots[0].reshape(len(plots[0]),)
            tmpy = plots[1].reshape(len(plots[1]),)
            x_coords = tmpx.tolist()
            y_coords = tmpy.tolist()
            x_coords.extend([x_max,x_min])
            y_coords.extend([y_max,y_min])
            x_max = max(x_coords)
            x_min = min(x_coords)
            y_max = max(y_coords)
            y_min = min(y_coords)
        x_max *= 1.2
        x_min *= 1.2
        y_max *= 1.2
        y_min *= 1.2
        return(x_max, x_min, y_max, y_min)
    except:
        print('error:find_xy_limits')

def Plot_Grid(Nodes, Lines):
    try:
        fig = plt.figure()
        ax = plt.subplot()
        bx = plt.subplot()

        nodes_in_zone = {}
        for z in Nodes.zone.unique():
            nodes_in_zone[z] = Nodes.index[Nodes.zone == z]

        for l in Lines.index:
            line_x = [Nodes.lat[Lines.source[l]], Nodes.lat[Lines.target[l]]]
            line_y = [Nodes.lon[Lines.source[l]], Nodes.lon[Lines.target[l]]]
            ann_x = sum(line_x)*0.5
            ann_y = sum(line_y)*0.5
            ax.plot(line_x,line_y, '-', c='k', linewidth=1)
            ax.annotate(l, (ann_x,ann_y), xytext=(ann_x*1.01,ann_y*1.01))

            for z in nodes_in_zone:
                x = []
                y = []
                for n in nodes_in_zone[z]:
                    x.append(Nodes.lat[n])
                    y.append(Nodes.lon[n])
                    #bx.annotate(n, (x[-1],y[-1]), xytext=(x[-1]*1.01,y[-1]*1.01))
                bx.scatter(x,y, marker='o', linewidths=42)
    except:
            print('error:Plot_Grid')


def from_named_rage(wb, name):
    position_obj = wb.name_map[name]
    address = position_obj[0].__dict__['formula_text'].split('!')
    sheet = wb.sheet_by_name(address[0])
    rng = address[1][1:].replace(":", "").split('$')
    dic = {}
    header = sheet.row_values(int(rng[1])-1, start_colx=a2i(rng[0]), end_colx=a2i(rng[2]))
    for idx,r in enumerate(range(int(rng[1]), int(rng[3]))):
        tmp = sheet.row_values(r, start_colx=a2i(rng[0]), end_colx=a2i(rng[2]))
        dic[idx] = {header[i]: tmp[i] for i in range(0,len(tmp))}
        df = pd.DataFrame.from_dict(dic, orient='index')
    return(df)


def a2i(alph): # alph_to_index_from_excelrange
    import numpy as np
    # A = 0 .... Z = 25, AA=26
    n = len(alph)
    p = 0
    for i in range(0, n):
        p += (ord(alph[i]) - ord('A') + 1) * np.power(26,(n-1-i))
    return(int(p-1))

def array2latex(array):
    for i in array:
        row = []
        for j in i:
            row.append(str(round(j,3)))

        print(' & '.join(row) + ' \\\ ')

def plot_chp_usage(gamsdb):
    for u in gamsdb["chp"]:
        G = []
        H = []
        for t in gamsdb["t"]:
            G.append(gamsdb["G"].find_record(keys=[u.get_keys()[0],t.get_keys()[0]]).level)
            H.append(gamsdb["H"].find_record(keys=[u.get_keys()[0],t.get_keys()[0]]).level)

        plt.figure()
        ax = plt.subplot()
        ax.set_title(u.get_keys()[0])
        ax.scatter(H,G)

def plot_es_use(gamsdb):
    # Beware ugly colors
#    gamsdb = model.out_db
    for u in gamsdb["es"]:
        G = []
        D = []
        L = []
        idx = []
        for i,t in enumerate(gamsdb["t"]):
            G.append(gamsdb["G"].find_record(keys=[u.get_keys()[0],t.get_keys()[0]]).level)
            D.append(gamsdb["D_es"].find_record(keys=[u.get_keys()[0],t.get_keys()[0]]).level)
            L.append(gamsdb["L_es"].find_record(keys=[u.get_keys()[0],t.get_keys()[0]]).level)
            idx.append(i)
        plt.figure()
        ax = plt.subplot()
        ax.set_title(u.get_keys()[0])
        ax.plot(idx, G, c='k', label="Storage Generation")
        ax.plot(idx, D, c='b', label="Storage Demand")
        ax.plot(idx, L, c='r', label="Sorage Level")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)


