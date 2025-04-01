from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
#from omniglot_dataset import OmniglotDataset
from protonet import ProtoNet
from data_loader import CustomDataset
from parser_util import get_parser

from tqdm import tqdm
import numpy as np
import torch
import os


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt, mode):
    # This dataset contains a long list of image and label pairings. The image size was [1,28,28], and the label is an int 
    # dataset = OmniglotDataset(mode=mode, root=opt.dataset_root)
    data_file = 'document_vectors_with_labels.pkl'
    new_data = CustomDataset(data_file)

    # print("Omniglot stuff:")
    # print("Number of items in the dataset:", type(dataset))
    # print("Dataset/image type:", dataset[0])
    # image, label = dataset[0]
    # print("Shape of the image tensor: ", image.shape)
    # print("This is the Label : ", label)

    # print("Our data:")
    # print("This is the type of labels ", type(new_data.labels), len(new_data.labels))
    # print("This is the type of vectors", type(new_data.vectors))
    # print("These is the vector at index 0:", new_data.vectors[0])
    # print("This is the label of vector at 0:", new_data.labels_idx[0])
    # print("This is the type of the vector at 0::", type(new_data.vectors[0]))

    # n_classes = len(np.unique(dataset.y))
    new_classes = len(np.unique(new_data.labels))


    
    if new_classes < opt.classes_per_it_tr or new_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return new_data
    # return dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr #currently set at 8
        num_samples = opt.num_support_tr + opt.num_query_tr #currently set at 5 and 5
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)#currently set at 100


def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    # sampler = init_sampler(opt, dataset.y, mode)
    sampler = init_sampler(opt,dataset.labels_idx, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler) 
    return dataloader


def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    input_dim = 50  # Fix Me: Update this with the actual size of your 1D vectors
    model = ProtoNet(input_dim=input_dim).to(device)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            #Current shape of y: ([80])
            #Current shape of x: ([80, 1, 50]) WHICH IS OKAY, WE HAVE NOT GOTTEN THE PREDICTIONS YET
            
            model_output = model(x) #CHECK THE OUTPUT HERE


            # print("this is the model output", model_output,"this is the model size:", model_output.shape)
            #FIX ME: Ensure the loss function matches new output shape
            loss, acc = loss_fn(model_output, target=y, n_support=opt.num_support_tr)
            #print("7")
            loss.backward()
            #print("8")
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            # FIX ME: Ensure the loss function matches your new output shape
            loss, acc = loss_fn(model_output, target=y, n_support=opt.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


# def test(opt, test_dataloader, model):
#     '''
#     Test the model trained with the prototypical learning algorithm
#     '''
#     device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
#     avg_acc = list()
#     for epoch in range(10):
#         test_iter = iter(test_dataloader)
#         for batch in test_iter:
#             x, y = batch
#             x, y = x.to(device), y.to(device)
#             model_output = model(x)
#             _, acc = loss_fn(model_output, target=y,
#                              n_support=opt.num_support_val)
#             avg_acc.append(acc.item())
#     avg_acc = np.mean(avg_acc)
#     print('Test Acc: {}'.format(avg_acc))

#     return avg_acc


# def eval(opt):
#     '''
#     Initialize everything and train
#     '''
#     options = get_parser().parse_args()

#     if torch.cuda.is_available() and not options.cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#     init_seed(options)
#     test_dataloader = init_dataset(options)[-1]
#     model = init_protonet(options)
#     model_path = os.path.join(opt.experiment_root, 'best_model.pth')
#     model.load_state_dict(torch.load(model_path))

#     test(opt=options,
#          test_dataloader=test_dataloader,
#          model=model)


def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)


    tr_dataloader = init_dataloader(options, 'train')
    val_dataloader = init_dataloader(options, 'val')
    trainval_dataloader = init_dataloader(options, 'trainval')
    test_dataloader = init_dataloader(options, 'test')

    
    model = init_protonet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    print("END!\n", best_state,"\n", best_acc,"\n", train_loss,"\n", train_acc,"\n", val_loss,"\n", val_acc)





    # USED FOR TESTING ONLY
    # print('Testing with last model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)

    # model.load_state_dict(best_state)
    # print('Testing with best model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)

    # optim = init_optim(options, model)
    # lr_scheduler = init_lr_scheduler(options, optim)

    # print('Training on train+val set..')
    # train(opt=options,
    #       tr_dataloader=trainval_dataloader,
    #       val_dataloader=None,
    #       model=model,
    #       optim=optim,
    #       lr_scheduler=lr_scheduler)

    # print('Testing final model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)


if __name__ == '__main__':
    main()

#  Omniglot data set
# Avg Train Loss: 0.028224853876745327, Avg Train Acc: 0.9896000069379807
# Avg Val Loss: 0.007427273888341652, Avg Val Acc: 0.9978666687011719 (Best: 0.9984000015258789)
# Testing with last model..
# Test Acc: 0.9960000029802323
# Testing with best model..
# Test Acc: 0.9957466698288917