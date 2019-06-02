from . import tasks

filename = 'controversial-comments'

'''
workflow = {
    'tasks': [
        (task1, task2),
        (task2, task3),
        (task3, task4)
    ],
    'params': {
        'param1': filename
    }
}
'''

def main():
    # Perform Task 1
    try:
        tasks.task1(filename)

    except:
        print("Task 1 Failed")

    else:

        # If Task 1 finishes, Perform Task 2
        try:
            tasks.task2(filename)

        except:
            print("Task 2 Failed")

        else:

            # If Task 2 finishes, Perform Task 3
            try:
                tasks.task3(filename)

            except:
                print("Task 3 Failed")

            else:

                # If Task 3 finishes, Perform Task 4
                try:
                    tasks.task4(filename)

                except:
                    print("Task 4 Failed")

                else:
                    print("All tasks finished.")

if __name__ == "__main__": main()
