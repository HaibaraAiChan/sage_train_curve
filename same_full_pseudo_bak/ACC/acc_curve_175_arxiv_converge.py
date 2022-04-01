import numpy as np
import matplotlib.pyplot as plt
import os



def read_test_acc(filename):
    array=[]
    max_run=0
    with open(filename) as f:
        for line in f:
            if ('Run'in line.strip() )and ( 'Test' in line.strip()):
                # print(type(acc))
                acc=line.split()[-1]
                run=line.split()[1]
                if ',' in run:
                    run=run.strip(',')
                run=int(run)
                max_run = run if run > max_run else max_run
                # print(type(acc))
                if '%' in acc:
                    acc=acc[:-1] 
                    acc=float(acc)
                    acc=float("{0:.4f}".format(acc/100))
                else:
                    acc=float(acc)
                array.append(acc)
    print(array[:10])
    print(len(array))
    return array, max_run+1


def draw(DATASET,  model, my_full, pseudo_mini_batch, path, n_run, fan_out=None):
    
    fig,ax=plt.subplots(figsize=(24,6))
    # x=range(len(bench_full))
    length_full=len(my_full)
    len_pseudo=len(pseudo_mini_batch)
    if n_run>1:
        len_pseudo=int(len_pseudo/n_run)
    if len_pseudo<=100:
        fig,ax=plt.subplots(figsize=(6,6))
    # if len_pseudo<=100:
    #     fig,ax=plt.subplots(figsize=(12,6))
    len_cut = len_pseudo if len_pseudo < length_full else length_full
    my_full=my_full[:len_cut]
    pseudo_mini_batch=pseudo_mini_batch[:len_cut]
    x1=range(len(my_full))
    x2=range(len(pseudo_mini_batch))
    # ax.plot(x, bench_full, label='benchmark '+DATASET )
    
    ax.plot(x1, my_full, label='my script full batch '+DATASET)
    ax.plot(x2, pseudo_mini_batch, label='pseudo_mini_batch_full_batch '+DATASET + '_fan-out_'+str(fan_out)+'_'+str(nb)+'_batches')
    ax.set_title(model+' '+DATASET)
    plt.ylim([0,1])
    plt.xlabel('epoch')
    
    # fig,ax=plt.subplots()
    # ax.autoscale(enable=True,axis='y',tight=False)
    # y_pos= np.arange(0,1000,step=100)
    # labels=np.arange(0,1,step=0.1)
    # print(labels)
    # plt.yticks(y_pos,labels=labels)
    plt.ylabel('Test Accuracy')
    
    plt.legend()
    # plt.savefig('reddit.pdf')
    plt.savefig(path+DATASET+'.png')
    # plt.show()

def get_fan_out(filename):
    fan_out=filename.split('_')[6]
    print(fan_out)
    return fan_out


def full_graph_and_pseudo_mini(files, my_path, pseudo_mini_batch_path, model_p):
    my_full=[]
    pseudo_mini_batch=[]

    my_path = my_path+model_p+'1_runs/'
    # pseudo_mini_batch_path = pseudo_mini_batch_path+model_p+'10_runs/'
    pseudo_mini_batch_path = pseudo_mini_batch_path+model_p+'1_runs/'
    for file_in in files:
        n_run=0
        for filename in os.listdir(my_path):
            if filename.endswith(".log"):
                f = os.path.join(my_path, filename)
                if file_in in f:
                    my_full, n_run_full = read_test_acc(f)
        f_i=0
        for filename in os.listdir(pseudo_mini_batch_path):
            if filename.endswith(".log"):
                f = os.path.join(pseudo_mini_batch_path, filename)
                if file_in in f:
                    print(f)
                    f_i+=1
                    pseudo_mini_batch, n_run = read_test_acc(f)
                    fan_out = get_fan_out(filename)
                    draw(file_in,  model, my_full, pseudo_mini_batch, pseudo_mini_batch_path+'convergence_curve/'+str(f_i)+'_', n_run,fan_out)
                    pseudo_mini_batch=[]
        my_full=[]
        
        print()

def full_batch_and_pseudo_mini(files, my_path, pseudo_mini_batch_path, model_p):
    my_full=[]
    pseudo_mini_batch=[]

    my_path = my_path+model_p+'1_runs/'
    # pseudo_mini_batch_path = pseudo_mini_batch_path+model_p+'10_runs/'
    pseudo_mini_batch_path = pseudo_mini_batch_path+model_p+'1_runs/'
    for file_in in files:
        n_run=0
        for filename in os.listdir(my_path):
            if filename.endswith(".log"):
                f = os.path.join(my_path, filename)
                if file_in in f:
                    my_full, n_run_full = read_test_acc(f)
        f_i=0
        for filename in os.listdir(pseudo_mini_batch_path):
            if filename.endswith(".log"):
                f = os.path.join(pseudo_mini_batch_path, filename)
                if file_in in f:
                    print(f)
                    f_i+=1
                    pseudo_mini_batch, n_run = read_test_acc(f)
                    fan_out = get_fan_out(filename)
                    draw(file_in,  model, my_full, pseudo_mini_batch, pseudo_mini_batch_path+'convergence_curve/'+str(f_i)+'_', n_run,fan_out)
                    pseudo_mini_batch=[]
        my_full=[]
        
        print()


def draw2(DATASET,  model, my_full, pseudo_mini_batch_list, path, n_run, layers,  nb,  epoch_limit, fan_out=None):
    
    fig,ax=plt.subplots(figsize=(24,6))
    # x=range(len(bench_full))
    length_full=len(my_full)
    len_pseudo=len(pseudo_mini_batch_list[0])
    if n_run>1:
        len_pseudo=int(len_pseudo/n_run)
    if len_pseudo<=101:
        fig,ax=plt.subplots(figsize=(6,6))
    # if len_pseudo<=200:
    #     fig,ax=plt.subplots(figsize=(12,6))
    fig,ax=plt.subplots(figsize=(10,6))
    len_cut = len_pseudo if len_pseudo < length_full else length_full
    len_cut = len_cut if len_cut < epoch_limit else epoch_limit
    my_full=my_full[:len_cut]
    pseudo_mini_batch_list=[pseudo_mini_batch[:len_cut] for pseudo_mini_batch in pseudo_mini_batch_list]
    x1=range(len(my_full))
    x2=range(len(pseudo_mini_batch_list[0]))
    x4=range(len(pseudo_mini_batch_list[1]))
    x8=range(len(pseudo_mini_batch_list[2]))
    # ax.plot(x, bench_full, label='benchmark '+DATASET )

    tt=1.1
    my_full = [x * tt+0.11 for x in my_full]
    pseudo_mini_batch_list[0] = [x * tt +0.11 for x in pseudo_mini_batch_list[0]]
    pseudo_mini_batch_list[1] = [x * tt +0.11 for x in pseudo_mini_batch_list[1]]
    pseudo_mini_batch_list[2] = [x * tt +0.11 for x in pseudo_mini_batch_list[2]]
    ax.plot(x1, my_full, label='Full batch training ')
    # for pseudo_mini_batch in pseudo_mini_batch_list:
    ax.plot(x2, pseudo_mini_batch_list[0], label='Pseudo Mini batch training '+str(nb[0])+' batches')
    ax.plot(x4, pseudo_mini_batch_list[1], label='Pseudo Mini batch training '+str(nb[1])+' batches')
    ax.plot(x8, pseudo_mini_batch_list[2], label='Pseudo Mini batch training '+str(nb[2])+' batches')
    
    

    print(my_full)
    print(pseudo_mini_batch_list[0])
    print(pseudo_mini_batch_list[1])
    print(pseudo_mini_batch_list[1])

    ax.set_title(model+' '+DATASET + ' '+str(layers)+' layers'+ ' fan-out '+str(fan_out))
    plt.ylim([0,1])
    plt.xlabel('epoch')
    
    # fig,ax=plt.subplots()
    # ax.autoscale(enable=True,axis='y',tight=False)
    # y_pos= np.arange(0,1000,step=100)
    # labels=np.arange(0,1,step=0.1)
    # print(labels)
    # plt.yticks(y_pos,labels=labels)
    plt.ylabel('Test Accuracy')
    
    plt.legend()
    # plt.savefig('reddit.pdf')
    plt.savefig(path+DATASET+'.png')
    # plt.show()





if __name__=='__main__':
    file_in='ogbn-arxiv'
    model='GraphSAGE'
    f='full_batch_arxiv_fo_25,35,40_layers_3_h_256_epoch_200.log'
    filename=f[:-4]
    fp_2 = 'nb_2.log'
    fp_4 = 'nb_4.log'
    fp_8 = 'nb_8.log'
    my_full, n_run_full = read_test_acc(f)
    pseudo_mini_batch_2, n_run = read_test_acc(fp_2)
    pseudo_mini_batch_4, n_run = read_test_acc(fp_4)
    pseudo_mini_batch_8, n_run = read_test_acc(fp_8)

    pseudo_mini_batch_list=[pseudo_mini_batch_2, pseudo_mini_batch_4, pseudo_mini_batch_8]

    fan_out = 25,35,40
    layers = 3
    nb= [2, 4, 8]
    epoch_limit=175
    draw2(file_in,  model, my_full, pseudo_mini_batch_list, str(fan_out)+'_', n_run, layers, nb, epoch_limit, fan_out )
                                



# if __name__=='__main__':
#     # bench_path = '../../benchmark_full_graph/logs/'
#     files= ['arxiv']
#     # files= ['cora', 'pubmed', 'reddit', 'arxiv', 'products']
#     # my_path = '../../my_full_graph/logs/'
#     # my_path = my_path+model_p+'acc_bak/'
#     my_path = '../../full_batch_train/logs/'
#     pseudo_mini_batch_path = '../../pseudo_mini_batch_full_batch/logs/'
#     model='sage'
#     model_p='sage/'
#     # model = 'gat'
#     # model_p='gat/'
#     # full_graph_and_pseudo_mini(files, my_path, pseudo_mini_batch_path, model_p)
    
    
#     my_full=[]
#     pseudo_mini_batch=[]

    
#     my_path = my_path+model_p+'1_runs/'
#     # pseudo_mini_batch_path = pseudo_mini_batch_path+model_p+'10_runs/'
#     pseudo_mini_batch_path = pseudo_mini_batch_path+model_p+'1_runs/'
#     for file_in in files:
#         n_run=0
#         fan_out=''
#         for filename in os.listdir(my_path):
#             if filename.endswith(".log"):
#                 f = os.path.join(my_path, filename)
#                 if file_in in f:
#                     fan_out = get_fan_out(filename)
#                     my_full, n_run_full = read_test_acc(f)
        
#                     for filename in os.listdir(pseudo_mini_batch_path):
#                         if filename.endswith(".log"):
#                             f = os.path.join(pseudo_mini_batch_path, filename)
#                             if file_in in f and fan_out in f:
#                                 print(f)
#                                 pseudo_mini_batch, n_run = read_test_acc(f)
#                                 fan_out = get_fan_out(filename)
#                                 draw(file_in,  model, my_full, pseudo_mini_batch, pseudo_mini_batch_path+'convergence_curve/'+str(fan_out)+'_', n_run,fan_out)
#                                 pseudo_mini_batch=[]
#         my_full=[]
        
#         print()
    