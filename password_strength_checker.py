def test_password_strength(password):
    # Criteria
    length = len(password)
    has_upper = any(char.isupper() for char in password)
    has_lower = any(char.islower() for char in password)
    has_digit = any(char.isdigit() for char in password)
    special_chars = set("!@#$%^&*()_+{}[];:'<>,./?")
    has_special = any(char in special_chars for char in password)

    # Strength calculation
    strength = 0
    if length >= 8:
        strength += 1
    if has_upper:
        strength += 1
    if has_lower:
        strength += 1
    if has_digit:
        strength += 1
    if has_special:
        strength += 1

    return strength

# Get user input for password
password_input = input("Enter your password: ")

# Test password strength
strength = test_password_strength(password_input)

if strength == 5:
    print("Strong password! 👍")
elif strength >= 3:
    print("Moderate password, consider improving. 👌")
else:
    print("Weak password. Please choose a stronger one. 👎")
