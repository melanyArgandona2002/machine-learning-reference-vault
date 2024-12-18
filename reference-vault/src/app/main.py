from capstone.mpg_regression import MpgRegressionCapstone
from capstone.tumor_classification import TumorClassificationCapstone
from capstone.lenet_5_character_recognition import CharacterRecognitionCapstone


def main() -> None:
    #mpg_capstone = MpgRegressionCapstone()
    #mpg_capstone.run()

    #tumor_capstone = TumorClassificationCapstone()
    #tumor_capstone.run()

    character_recognition = CharacterRecognitionCapstone()
    character_recognition.run()    

if __name__ == "__main__":
    main()
