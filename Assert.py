
def PrintAssertRegression(score, difficulty) :
    if score < 1/180 :
        successful(difficulty)
    else:
        fail(difficulty)
    print(difficulty + " test mean absolute error: ", score)

def PrintAssertClassification(score, difficulty) :
    if score > 0.98 :
        successful(difficulty)
    else:
        fail(difficulty)
    print(difficulty + " test accuracy: ", score)


def successful(difficulty):
    print("==============================")
    print("=== " + difficulty.upper() + " TEST SUCCESSFUL ===")
    print("==============================")

def fail(difficulty):
        print("==============================")
        print("======= " + difficulty.upper() + " TEST FAIL =======")
        print("==============================")
