class Configuration(object):
    hum_njoints = 22  # joint number of Lafan1 dataset (exclude End sites)
    dog_njoints = 21  # joint number of Dog dataset (exclude End sites)

    # The value in correspondence are indices in the skeleton tree excluding the End sites.
    correspondence = [{"hum_joints": [3, 4, 7, 8, 1, 2, 5, 6], "dog_joints": [8, 15, 12, 18, 5, 6, 7, 9, 10, 11, 13, 14, 16, 17]}, # two/four legs
                      {"hum_joints": [9, 10, 11], "dog_joints": [1, 2]}, # Spine
                      {"hum_joints": [12, 13], "dog_joints": [3, 4]}]  # Head

    dog_end = [10, 15, 19, 23]   # These value are indices of the used End Sites in Dog skeleton
    hum_end = [5, 10, 16]        # These value are indices of the used End Sites in human skeleton