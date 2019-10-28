
def PrintAssertRegression(score, difficulty) :
    if score < 1/360 : # Teleportation has a cooldown of 360 sencodes.
        successful(difficulty)
    else:
        fail(difficulty)
    print(difficulty + " test mean absolute error: ", score)

def PrintAssertClassification(score, difficulty) :
    if score > 0.98 : # Teleportation has a cooldown of 360 sencodes.
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
