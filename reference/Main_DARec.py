import sys, os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')

from DARec import DARec
import tensorflow as tf

# import RecEval as evl
# import MatUtils as mtl
import Utils.GenUtils as gtl
# import ModUtils as mod

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Loading Data from Hard Drive
path_dict = {'amazon-books-elecs':
             ('Data/Amazon_Shared_User/Books_and_Electronics/Books_Data.mat',
              'Data/Amazon_Shared_User/Books_and_Electronics/Electronics_Data.mat',
              'Data/Amazon_Shared_User/Books_and_Electronics/AmazonBooks-ebd-org-500.mat',
              'Data/Amazon_Shared_User/Books_and_Electronics/AmazonElectronics-ebd-org-500.mat'),

             'amazon-mtv-office':
             ('Data/Amazon_Shared_User/MTV_and_Office/MTV_Data.mat',
              'Data/Amazon_Shared_User/MTV_and_Office/Office_Data.mat',
              'Data/Amazon_Shared_User/MTV_and_Office/AmazonMTV-ebd-org-500.mat',
              'Data/Amazon_Shared_User/MTV_and_Office/AmazonOffice-ebd-org-500.mat',
              'Data/Amazon_Shared_User/MTV_and_Office/AmazonMTV-BPMF-ebd-org-800.mat',
              'Data/Amazon_Shared_User/MTV_and_Office/AmazonOffice-BPMF-ebd-org-800.mat'
             ),

             'amazon-beauty-clothing':
             ('Data/Amazon_Shared_User/Beauty_and_Clothing_Shoes_and_Jewelry/Beauty_Data.mat',
              'Data/Amazon_Shared_User/Beauty_and_Clothing_Shoes_and_Jewelry/Clothing_Shoes_and_Jewelry_Data.mat',
              'Data/Amazon_Shared_User/Beauty_and_Clothing_Shoes_and_Jewelry/AmazonBeauty-ebd-org-800.mat',
              'Data/Amazon_Shared_User/Beauty_and_Clothing_Shoes_and_Jewelry/AmazonClothing-ebd-org-800.mat'
             ),

            'amazon-automotive-toys':
             (
              'Data/Amazon_Shared_User/Automotive_and_Toys_and_Games/Automotive_Data.mat',
              'Data/Amazon_Shared_User/Automotive_and_Toys_and_Games/Toys_and_Games_Data.mat',
              'Data/Amazon_Shared_User/Automotive_and_Toys_and_Games/AmazonAutomotive-ebd-org-1000.mat',
              'Data/Amazon_Shared_User/Automotive_and_Toys_and_Games/AmazonToys-ebd-org-1000.mat',
             ),

            'amazon-books-baby':
             (
              'Data/Amazon_Shared_User/Books_and_Baby/Books_Data.mat',
              'Data/Amazon_Shared_User/Books_and_Baby/Baby_Data.mat',
              'Data/Amazon_Shared_User/Books_and_Baby/AmazonBooks-ebd-org-1000.mat',
              'Data/Amazon_Shared_User/Books_and_Baby/AmazonBaby-ebd-org-1000.mat',
             ),

             'amazon-cds-sports':
             (
                'Data/Amazon_Shared_User/CDs_and_Vinyl_and_Sports_and_Outdoors/CDs_and_Vinyl_Data.mat',
                'Data/Amazon_Shared_User/CDs_and_Vinyl_and_Sports_and_Outdoors/Sports_and_Outdoors_Data.mat',
                'Data/Amazon_Shared_User/CDs_and_Vinyl_and_Sports_and_Outdoors/AmazonCDs-ebd-org-800.mat',
                'Data/Amazon_Shared_User/CDs_and_Vinyl_and_Sports_and_Outdoors/AmazonSports-ebd-org-800.mat',
                'Data/Amazon_Shared_User/CDs_and_Vinyl_and_Sports_and_Outdoors/AmazonCDs-BPMF-ebd-org-150.mat',
                'Data/Amazon_Shared_User/CDs_and_Vinyl_and_Sports_and_Outdoors/AmazonSports-BPMF-ebd-org-150.mat'
             ),

            'amazon-hk-kindle':
             (
                'Data/Amazon_Shared_User/Home_and_Kitchen_and_Kindle_Store/Kindle_Store_Data.mat',
                'Data/Amazon_Shared_User/Home_and_Kitchen_and_Kindle_Store/Home_and_Kitchen_Data.mat',
                'Data/Amazon_Shared_User/Home_and_Kitchen_and_Kindle_Store/AmazonKindle-ebd-org-1000.mat',
                'Data/Amazon_Shared_User/Home_and_Kitchen_and_Kindle_Store/AmazonHomeKitchens-ebd-org-1000.mat',
             ),

            'amazon-elec-grocery':
             (
                'Data/Amazon_Shared_User/Electronics_and_Grocery_and_Gourmet_Food/Grocery_and_Gourmet_Food_Data.mat',
                'Data/Amazon_Shared_User/Electronics_and_Grocery_and_Gourmet_Food/Electronics_Data.mat',
                'Data/Amazon_Shared_User/Electronics_and_Grocery_and_Gourmet_Food/AmazonGrocery-ebd-org-800.mat',
                'Data/Amazon_Shared_User/Electronics_and_Grocery_and_Gourmet_Food/AmazonElectronics-ebd-org-800.mat',
             ),

             'amazon-pet-music':
             (
                'Data/Amazon_Shared_User/Pet_Supplies_and_Digital_Music/Digital_Music_Data.mat',
                'Data/Amazon_Shared_User/Pet_Supplies_and_Digital_Music/Pet_Supplies_Data.mat',
                'Data/Amazon_Shared_User/Pet_Supplies_and_Digital_Music/AmazonMusic-ebd-org-500.mat',
                'Data/Amazon_Shared_User/Pet_Supplies_and_Digital_Music/AmazonPet-ebd-org-500.mat',
             ),

            'amazon-game-app':
             (
                'Data/Amazon_Shared_User/Video_Games_and_Apps_for_Android/Video_Games_Data.mat',
                'Data/Amazon_Shared_User/Video_Games_and_Apps_for_Android/Apps_for_Android_Data.mat',
                'Data/Amazon_Shared_User/Video_Games_and_Apps_for_Android/AmazonGame-ebd-org-900.mat',
                'Data/Amazon_Shared_User/Video_Games_and_Apps_for_Android/AmazonApps-ebd-org-900.mat',
                'Data/Amazon_Shared_User/Video_Games_and_Apps_for_Android/AmazonGame-BPMF-ebd-org-100.mat',
                'Data/Amazon_Shared_User/Video_Games_and_Apps_for_Android/AmazonApps-BPMF-ebd-org-100.mat'
             ),

            'amazon-phone-health':
             (
                'Data/Amazon_Shared_User/Cell_Phones_and_Accessories_and_Health_and_Personal_Care/Cell_Phones_and_Accessories_Data.mat',
                'Data/Amazon_Shared_User/Cell_Phones_and_Accessories_and_Health_and_Personal_Care/Health_and_Personal_Care_Data.mat',
                'Data/Amazon_Shared_User/Cell_Phones_and_Accessories_and_Health_and_Personal_Care/AmazonCellPhone-ebd-org-500.mat',
                'Data/Amazon_Shared_User/Cell_Phones_and_Accessories_and_Health_and_Personal_Care/AmazonHealth-ebd-org-500.mat',
             ),

            'amazon-beauty-tool':
             (
                'Data/Amazon_Shared_User/Beauty_and_Tools_and_Home_Improvement/Beauty_Data.mat',
                'Data/Amazon_Shared_User/Beauty_and_Tools_and_Home_Improvement/Tools_and_Home_Improvement_Data.mat',
                'Data/Amazon_Shared_User/Beauty_and_Tools_and_Home_Improvement/AmazonBeauty-ebd-org-300.mat',
                'Data/Amazon_Shared_User/Beauty_and_Tools_and_Home_Improvement/AmazonTool-ebd-org-300.mat',
             ),
            }

dataset = 'amazon-mtv-office'
# dataset = 'amazon-books-elecs'
# dataset = 'amazon-beauty-clothing'
# dataset = 'amazon-automotive-toys'
# dataset = 'amazon-books-baby'
# dataset = 'amazon-cds-sports'
# dataset = 'amazon-hk-kindle'
# dataset = 'amazon-elec-grocery'
# dataset = 'amazon-pet-music'
# dataset = 'amazon-game-app'
# dataset = 'amazon-phone-health'
# dataset = 'amazon-beauty-tool'

path_sc, path_tg, path_sc_ebd, path_tg_ebd, _, _ = path_dict[dataset]
# Test For Embedding generated from BPMF
# path_sc,path_tg, _, _, path_sc_ebd, path_tg_ebd = path_dict[dataset]

print('Loading Data From Source {0} and Target {1}'.format(path_sc, path_tg))
data_sc, data_tg = gtl.load_mat_as_matrix(path_sc, opt='coo'), gtl.load_mat_as_matrix(path_tg, opt='coo')
original_matrix_sc, train_matrix_sc, test_matrix_sc = data_sc['original'], data_sc['train'], data_sc['test']
original_matrix_tg, train_matrix_tg, test_matrix_tg = data_tg['original'], data_tg['train'], data_tg['test']

sc_ebd, tg_ebd = gtl.load_mat_as_array(path_sc_ebd), gtl.load_mat_as_array(path_tg_ebd)
embedding_arr_sc, embedding_arr_tg = sc_ebd['embedding'], tg_ebd['embedding']
# Test For Embedding generated from BPMF
# embedding_arr_sc, embedding_arr_tg = sc_ebd['user_array'], tg_ebd['user_array']
# embedding_arr_sc, embedding_arr_tg = sc_ebd['item_array'], tg_ebd['item_array']

# The Main Program
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                    # intra_op_parallelism_threads=24,
                                    # inter_op_parallelism_threads=24,
                                    gpu_options=gpu_options)) as sess:
    model = DARec(
        sess,
        training_mode='dann',
        input_dim=500,

        pred_dim=60,
        shared_dim=32,
        pred_sc_tg_lambda=0.1,    # Adjust the weight of the loss in the target domain
        pred_reg=1e-6,

        cls_layers=[16,8,2],
        cls_reg=1e-5,

        # pred_cls_lambda=5.0,    # Adjust the weight of the classification loss
        grl_lambda=0.1,
        drop_out_rate=0.25,

        dec_nn_dim_sc=500,
        dec_nn_dim_tg=500,

        # dec_nn_layers_sc = [200,800],
        # dec_nn_layers_tg = [200],

        domain_loss_ratio= 1,

        mode='user',
        # mode='item',

        lr=0.001,
        epochs=1500,
        batch_size=128,
        T=10 ** 3,
        verbose=True
    )

    model.prepare_data(original_matrix_sc, train_matrix_sc, test_matrix_sc,
                        original_matrix_tg, train_matrix_tg, test_matrix_tg,
                        embedding_arr_sc, embedding_arr_tg)

    model.build_model()
    model.train()
