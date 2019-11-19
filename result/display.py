
def regression(score, difficulty) :
    if score < 1/180 :
        __successful(difficulty)
    else:
        __fail(difficulty)
    print(difficulty + " score: ", score)

def classification(score, difficulty) :
    if score > 0.98 :
        __successful(difficulty)
    else:
        __fail(difficulty)
    print(difficulty + " score: ", score*100, "%")


def __successful(difficulty):
    print("==============================")
    print("=== " + difficulty.upper() + " TEST SUCCESSFUL ===")
    print("==============================")

def __fail(difficulty):
    print("==============================")
    print("======= " + difficulty.upper() + " TEST FAIL =======")
    print("==============================")
