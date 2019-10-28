
def PrintAssert(mae, difficulty) :
    if mae < 1/360 : # Teleportation has a cooldown of 360 sencodes.
        print("========================")
        print("==== " + difficulty.upper() + " TEST SUCCESSFUL ===")
        print("========================")
    else:
        print("========================")
        print("==== " + difficulty.upper() + " TEST FAIL ====")
        print("========================")
    print(difficulty + " test mean absolute error: ", mae)
