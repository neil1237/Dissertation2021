import warnings
warnings.filterwarnings("ignore")

print("Welcome to 'Understanding False Positive Fraud Detection In The Context Of AI Classification For Financial Transactions'")
print("Neil Mizzi SWD6.3A 2021")
input("Press Enter to continue...")
print("")

while True:
    try:
        print("Press 1 to Analyse the Dataset")
        print("Press 2 to Run Fraud Detection")
        print("Press 3 to Analyse the False Positive transactions within both returned datasets ")
        print("")
        userValue = int(input('Please select an option ..... '))

        if userValue == 1:
            print('Analysing Data')
            exec(open("1 Anyalsis Of Dataset.py").read())
            input("Press Enter to continue...")
            continue

        elif userValue == 2:
            print('Running fraud detection')
            exec(open("2 Model Creation and Training.py").read())
            input("Press Enter to continue...")
            continue
        elif userValue == 3:
            print('Analysing the false positives')
            exec(open("3 falsePositiveAnalysis.py").read())
            input("Press Enter to continue...")
            continue
        else:
            print('Please try again')

    except:
        print("That's not a valid option!")

