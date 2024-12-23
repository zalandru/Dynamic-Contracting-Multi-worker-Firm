import json

def load_assignments(file_name):
    """Load Secret Santa assignments from a JSON file."""
    try:
        with open(file_name, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
        return None

def reveal_assignment(assignments, name):
    """Reveal the Secret Santa assignment for the given name."""
    if name in assignments:
        print(f"{name}, you are assigned to: {assignments[name]}")
    else:
        print(f"No assignment found for {name}. Check the spelling or try again.")

def main():
    file_name = "secret_santa_assignments.json"
    print("Welcome to the Secret Santa assignment reveal!")
    assignments = load_assignments(file_name)

    if assignments:
        while True:
            name = input("Enter your name to see who you're assigned to (or type 'exit' to quit): ").strip()
            if name.lower() == "exit":
                print("Goodbye! Happy gifting!")
                break
            reveal_assignment(assignments, name)

if __name__ == "__main__":
    main()

