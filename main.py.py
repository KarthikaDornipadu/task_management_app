import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to load tasks from CSV
def load_tasks():
    try:
        return pd.read_csv('tasks.csv')
    except FileNotFoundError:
        return pd.DataFrame(columns=["description", "priority"])

# Function to save tasks to CSV
def save_tasks(tasks):
    tasks.to_csv('tasks.csv', index=False)

# Function to add a task
def add_task(tasks, description, priority):
    tasks = tasks.append({"description": description, "priority": priority}, ignore_index=True)
    save_tasks(tasks)
    print("Task added successfully.")

# Function to remove a task
def remove_task(tasks, description):
    tasks = tasks[tasks['description'] != description]
    save_tasks(tasks)
    print("Task removed successfully.")

# Function to list tasks
def list_tasks(tasks):
    print(tasks.to_string(index=False))

# Function to prioritize tasks (sort by priority)
def prioritize_tasks(tasks):
    priority_order = {"High": 1, "Medium": 2, "Low": 3}
    tasks['priority_num'] = tasks['priority'].map(priority_order)
    tasks = tasks.sort_values(by=['priority_num'])
    tasks = tasks.drop(columns=['priority_num'])
    save_tasks(tasks)
    print("Tasks prioritized successfully.")
    list_tasks(tasks)

# Function to recommend a task based on a description
def recommend_task(tasks, description):
    if tasks.empty:
        print("No tasks available for recommendation.")
        return
    vectorizer = TfidfVectorizer(stop_words='english')
    tasks_tfidf = vectorizer.fit_transform(tasks['description'])
    description_tfidf = vectorizer.transform([description])
    cosine_sim = cosine_similarity(description_tfidf, tasks_tfidf)
    recommended_index = cosine_sim.argmax()
    recommended_task = tasks.iloc[recommended_index]
    print(f"Recommended Task: {recommended_task['description']} (Priority: {recommended_task['priority']})")

# Main function to run the app
def main():
    tasks = load_tasks()

    while True:
        print("\nTask Management App")
        print("1. Add Task")
        print("2. Remove Task")
        print("3. List Tasks")
        print("4. Prioritize Tasks")
        print("5. Recommend Task Based on Description")
        print("6. Exit")
        
        choice = input("Choose an option: ")

        if choice == '1':
            description = input("Enter task description: ")
            priority = input("Enter task priority (High/Medium/Low): ")
            add_task(tasks, description, priority)
        elif choice == '2':
            description = input("Enter task description to remove: ")
            remove_task(tasks, description)
        elif choice == '3':
            list_tasks(tasks)
        elif choice == '4':
            prioritize_tasks(tasks)
        elif choice == '5':
            description = input("Enter task description for recommendation: ")
            recommend_task(tasks, description)
        elif choice == '6':
            print("Exiting the app.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()