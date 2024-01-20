import os
from resampleFolder import resample_folder
from filterData import filter_data_by_class, copy_files_to_folder_structure

RESAMPLING = True
RESAMPLING_IMAGE_SIZE = 256                     # 512, 256, 128
RESAMPLING_INPUT_FOLDER = './../Data50'
RESAMPLING_OUTPUT_FOLDER = f"./../Data50_resampled_{RESAMPLING_IMAGE_SIZE}"
if not(RESAMPLING):
    RESAMPLING_OUTPUT_FOLDER = RESAMPLING_INPUT_FOLDER

FILTERING = True
FILTERING_CLASS = 0                             # DEFAULT VALUE = 0 --> This value stands for the unlabeled class
FILTERING_THRESHOLD_FOR_CLASS = 20              # (0..100)
#FILTERING_INPUT_FOLDER_TRAIN = "./../Data50_resampled_{RESAMPLING_IMAGE_SIZE}/train/"
FILTERING_INPUT_FOLDER_TRAIN = os.path.join(RESAMPLING_OUTPUT_FOLDER, 'train')
FILTERING_OUTPUT_FOLDER_TRAIN = f"./../Data50_res{RESAMPLING_IMAGE_SIZE}_filtered_{FILTERING_THRESHOLD_FOR_CLASS}/train/"
#FILTERING_INPUT_FOLDER_VAL = "./../Data50_resampled_{RESAMPLING_IMAGE_SIZE}/validate/"
FILTERING_INPUT_FOLDER_VAL = os.path.join(RESAMPLING_OUTPUT_FOLDER, 'validate')
FILTERING_OUTPUT_FOLDER_VAL = f"./../Data50_res{RESAMPLING_IMAGE_SIZE}_filtered_{FILTERING_THRESHOLD_FOR_CLASS}/validate/"
#FILTERING_INPUT_FOLDER_TEST = "./../Data50_resampled_{RESAMPLING_IMAGE_SIZE}/test/"
FILTERING_INPUT_FOLDER_TEST = os.path.join(RESAMPLING_OUTPUT_FOLDER, 'test')
FILTERING_OUTPUT_FOLDER_TEST = f"./../Data50_res{RESAMPLING_IMAGE_SIZE}_filtered_{FILTERING_THRESHOLD_FOR_CLASS}/test/"


################################################
##############                    ##############
############      Main section      ############
##############                    ##############

if __name__ == "__main__":

    ##################################################
    #                                                #
    #                RESAMPLING                      #
    #                                                #

    if(RESAMPLING):
        TRAIN_SAT_FOLDER = os.path.join(RESAMPLING_INPUT_FOLDER, 'train/sat')
        TRAIN_GT_FOLDER = os.path.join(RESAMPLING_INPUT_FOLDER, 'train/gt')
        VALIDATE_SAT_FOLDER = os.path.join(RESAMPLING_INPUT_FOLDER, 'validate/sat')
        VALIDATE_GT_FOLDER = os.path.join(RESAMPLING_INPUT_FOLDER, 'validate/gt')
        TEST_SAT_FOLDER = os.path.join(RESAMPLING_INPUT_FOLDER, 'test/sat')
        TEST_GT_FOLDER = os.path.join(RESAMPLING_INPUT_FOLDER, 'test/gt')

        # Resample train data
        resample_folder(TRAIN_SAT_FOLDER, os.path.join(RESAMPLING_OUTPUT_FOLDER, 'train/sat'), target_size=(RESAMPLING_IMAGE_SIZE, RESAMPLING_IMAGE_SIZE))
        resample_folder(TRAIN_GT_FOLDER, os.path.join(RESAMPLING_OUTPUT_FOLDER, 'train/gt'), target_size=(RESAMPLING_IMAGE_SIZE, RESAMPLING_IMAGE_SIZE))

        # Resample validate data
        resample_folder(VALIDATE_SAT_FOLDER, os.path.join(RESAMPLING_OUTPUT_FOLDER, 'validate/sat'), target_size=(RESAMPLING_IMAGE_SIZE, RESAMPLING_IMAGE_SIZE))
        resample_folder(VALIDATE_GT_FOLDER, os.path.join(RESAMPLING_OUTPUT_FOLDER, 'validate/gt'), target_size=(RESAMPLING_IMAGE_SIZE, RESAMPLING_IMAGE_SIZE))

        # Resample test data
        resample_folder(TEST_SAT_FOLDER, os.path.join(RESAMPLING_OUTPUT_FOLDER, 'test/sat'), target_size=(RESAMPLING_IMAGE_SIZE, RESAMPLING_IMAGE_SIZE))
        resample_folder(TEST_GT_FOLDER, os.path.join(RESAMPLING_OUTPUT_FOLDER, 'test/gt'), target_size=(RESAMPLING_IMAGE_SIZE, RESAMPLING_IMAGE_SIZE))


    ##################################################
    #                                                #
    #                  FILTERING                     #
    #                                                #

    if(FILTERING):
        # Filter train data
        filtered_sat_files, filtered_gt_files = filter_data_by_class(FILTERING_INPUT_FOLDER_TRAIN, FILTERING_CLASS, FILTERING_THRESHOLD_FOR_CLASS)
        copy_files_to_folder_structure(filtered_sat_files, filtered_gt_files, FILTERING_OUTPUT_FOLDER_TRAIN)

        # Filter validate data
        filtered_sat_files, filtered_gt_files = filter_data_by_class(FILTERING_INPUT_FOLDER_VAL, FILTERING_CLASS, FILTERING_THRESHOLD_FOR_CLASS)
        copy_files_to_folder_structure(filtered_sat_files, filtered_gt_files, FILTERING_OUTPUT_FOLDER_VAL)

        # Filter test data
        filtered_sat_files, filtered_gt_files = filter_data_by_class(FILTERING_INPUT_FOLDER_TEST, FILTERING_CLASS, FILTERING_THRESHOLD_FOR_CLASS)
        copy_files_to_folder_structure(filtered_sat_files, filtered_gt_files, FILTERING_OUTPUT_FOLDER_TEST)