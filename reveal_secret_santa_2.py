import random

def generate_assignments(participants, constraints, seed=10000):
    """Generate consistent Secret Santa assignments with a fixed seed."""
    random.seed(seed)  # Set the seed for consistent results
    
    # Generate all possible pairings
    pairings = [(giver, receiver) for giver in participants for receiver in participants if giver != receiver]
    
    # Filter out invalid pairings based on constraints
    valid_pairings = [
        (giver, receiver) for giver, receiver in pairings
        if not (giver, receiver) in constraints and not (receiver, giver) in constraints
    ]
    
    # Helper function to find assignments
    def find_assignment(participants, valid_pairings, assignment=None):
        if assignment is None:
            assignment = {}
        if len(assignment) == len(participants):
            return assignment  # Found a complete assignment

        giver = next(p for p in participants if p not in assignment)
        possible_receivers = [
            receiver for _, receiver in valid_pairings
            if _ == giver and receiver not in assignment.values()
        ]

        random.shuffle(possible_receivers)  # Shuffle receivers for variety
        for receiver in possible_receivers:
            assignment[giver] = receiver
            if find_assignment(participants, valid_pairings, assignment):
                return assignment
            del assignment[giver]  # Backtrack
        
        return None  # No valid assignment found
    
    return find_assignment(participants, valid_pairings)

def reveal_assignment(assignments, name):
    """Reveal the Secret Santa assignment for the given name."""
    if name in assignments:
        print(f"{name}, you are assigned to: {assignments[name]}")
    else:
        print(f"No assignment found for {name}. Check the spelling or try again.")

def main():
    participants = ["Andrei", "Gaya", "Gosha", "Marsel", "Tonya", "Marina"]
    constraints = [
        ("Andrei", "Gaya"), ("Gaya", "Andrei"),
        ("Andrei", "Gosha"),
        ("Marsel", "Tonya"), ("Tonya", "Marsel")
    ]
    
    # Generate assignments with a fixed seed
    assignments = generate_assignments(participants, constraints, seed=42)
    
    if not assignments:
        print("Error: Could not generate valid assignments with the given constraints.")
        return
    
    print("Welcome to the Secret Santa assignment reveal!")
    while True:
        name = input("Enter your name to see who you're assigned to (or type 'exit' to quit): ").strip()
        if name.lower() == "exit":
            print("Goodbye! Happy gifting!")
            break
        reveal_assignment(assignments, name)

if __name__ == "__main__":
    main()
