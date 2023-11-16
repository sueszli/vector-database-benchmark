from utils_cv.action_recognition.model import VideoLearner

def test_VideoLearner(ar_milk_bottle_dataset) -> None:
    if False:
        i = 10
        return i + 15
    ' Test VideoLearner Initialization. '
    learner = VideoLearner(ar_milk_bottle_dataset, num_classes=2)
    learner.fit(lr=0.001, epochs=1)
    learner.evaluate()

def test_VideoLearner_using_split_file(ar_milk_bottle_dataset_with_split_file) -> None:
    if False:
        return 10
    ' Test VideoLearner Initialization. '
    learner = VideoLearner(ar_milk_bottle_dataset_with_split_file, num_classes=2)
    learner.fit(lr=0.001, epochs=1)
    learner.evaluate()