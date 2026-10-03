print("Wie heißen Sie?")
name = input()
print("Hallo, " + name + "!")
tasks = []

def add_task():
    task = input("Enter task: ")
    tasks.append({"task": task, "done": False})
    print("Task added.")

def delete_task():
    list_tasks()
    try:
        task_index = int(input("Enter task number to delete: ")) - 1
        del tasks[task_index]
        print("Task deleted.")
    except IndexError:
        print("Invalid task number.")

def mark_done():
    list_tasks()
    try:
        task_index = int(input("Enter task number to mark as done: ")) - 1
        tasks[task_index]["done"] = True
        print("Task marked as done.")
    except IndexError:
        print("Invalid task number.")

def list_tasks():
    if not tasks:
        print("No tasks in list.")
    else:
        for i, task in enumerate(tasks):
            status = "Done" if task["done"] else "Not Done"
            print(f"{i+1}. {task['task']} - {status}")

while True:
    print("\nOptions:")
    print("1. Add task")
    print("2. Delete task")
    print("3. Mark as done")
    print("4. List tasks")
    print("5. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        add_task()
    elif choice == "2":
        delete_task()
    elif choice == "3":
        mark_done()
    elif choice == "4":
        list_tasks()
    elif choice == "5":
        break
    else:
        print("Invalid choice.")
"Make a simple to-do list app where I can add, delete, and mark tasks as done."