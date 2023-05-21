from torchvision import datasets
from torchvision import transforms

# Save data path
save_dir = "~/data/"

# Data Augmentation
base_train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)

base_test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# Data Information
data_map = {
    "cifar10" : {
        '#class' : 10,
        'dataset' : datasets.CIFAR10
    },
    'cifar100' : {
        '#class' : 100,
        'dataset' : datasets.CIFAR100
    },
    'imagenet' : {
        '#class' : 1000,
        'dataset' : datasets.ImageNet
    }
}

# Get Dataset
def get_dataset(dataset_name:str, train_transform = base_train_transform, test_transform = base_test_transform):
    if dataset_name not in list(data_map.keys()):
        raise Exception(f"The data set {dataset_name} is currently not supported")
    data_info = data_map[dataset_name]
    class_cnt = data_info['#class']
    
    if dataset_name == 'imagenet':
        train_dataset = data_info['dataset'](
            root = save_dir,
            transform = train_transform,
            split = 'train'
        )
        test_dataset = data_info['dataset'](
            root = save_dir,
            transform = test_transform,
            split = 'val'
        )
        
    else:
        train_dataset = data_info['dataset'](
            root = save_dir,
            transform = train_transform,
            train = True,
            download = True
        )
        test_dataset = data_info['dataset'](
            root = save_dir,
            transform = test_transform,
            train = False,
            download = True
        )
    return (class_cnt, train_dataset, test_dataset)