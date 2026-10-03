# ============================================================
# TIC-TAC-TOE USING MINIMAX ALGORITHM
# Human = X
# AI    = O
# Board = 3 x 3
# Interface = Command Line
# ============================================================

import math

# ------------------------------------------------------------
# GAME STATISTICS
# ------------------------------------------------------------

human_wins = 0
ai_wins = 0
draws = 0


# ------------------------------------------------------------
# DISPLAY BOARD
# ------------------------------------------------------------

def display_board(board):
    print("\n")
    print("     |     |")
    print(f"  {board[0]}  |  {board[1]}  |  {board[2]}")
    print("_____|_____|_____")
    print("     |     |")
    print(f"  {board[3]}  |  {board[4]}  |  {board[5]}")
    print("_____|_____|_____")
    print("     |     |")
    print(f"  {board[6]}  |  {board[7]}  |  {board[8]}")
    print("     |     |")
    print("\n")


# ------------------------------------------------------------
# DISPLAY POSITION NUMBERS
# ------------------------------------------------------------

def display_position_guide():
    print("\nPosition Guide:")
    print("""
     1  |  2  |  3
    ----+-----+----
     4  |  5  |  6
    ----+-----+----
     7  |  8  |  9
    """)


# ------------------------------------------------------------
# CHECK WINNER
# ------------------------------------------------------------

def check_winner(board):
    winning_combinations = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6)
    ]

    for a, b, c in winning_combinations:

        if board[a] == board[b] == board[c]:

            if board[a] == "X":
                return "X"

            if board[a] == "O":
                return "O"

    # If no empty spaces remain, it is a draw
    if all(space != " " for space in board):
        return "Draw"

    return None


# ------------------------------------------------------------
# GET AVAILABLE MOVES
# ------------------------------------------------------------

def get_available_moves(board):
    return [i for i in range(9) if board[i] == " "]


# ------------------------------------------------------------
# MINIMAX ALGORITHM
# ------------------------------------------------------------

def minimax(board, depth, is_maximizing):

    result = check_winner(board)

    # AI wins
    if result == "O":
        return 10 - depth

    # Human wins
    if result == "X":
        return depth - 10

    # Draw
    if result == "Draw":
        return 0

    # --------------------------------------------------------
    # AI'S TURN - MAXIMIZE SCORE
    # --------------------------------------------------------

    if is_maximizing:

        best_score = -math.inf

        for move in get_available_moves(board):

            board[move] = "O"

            score = minimax(
                board,
                depth + 1,
                False
            )

            board[move] = " "

            best_score = max(best_score, score)

        return best_score

    # --------------------------------------------------------
    # HUMAN'S TURN - MINIMIZE SCORE
    # --------------------------------------------------------

    else:

        best_score = math.inf

        for move in get_available_moves(board):

            board[move] = "X"

            score = minimax(
                board,
                depth + 1,
                True
            )

            board[move] = " "

            best_score = min(best_score, score)

        return best_score


# ------------------------------------------------------------
# FIND BEST MOVE FOR AI
# ------------------------------------------------------------

def find_best_move(board):

    best_score = -math.inf
    best_move = None

    for move in get_available_moves(board):

        board[move] = "O"

        score = minimax(
            board,
            0,
            False
        )

        board[move] = " "

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


# ------------------------------------------------------------
# HUMAN MOVE
# ------------------------------------------------------------

def human_move(board):

    while True:

        try:

            position = int(
                input("Enter your move (1-9): ")
            )

            # Convert position to list index
            position -= 1

            if position < 0 or position > 8:
                print("Please enter a number between 1 and 9.")
                continue

            if board[position] != " ":
                print("That position is already occupied.")
                continue

            board[position] = "X"
            break

        except ValueError:

            print("Invalid input. Please enter a number from 1 to 9.")


# ------------------------------------------------------------
# AI MOVE
# ------------------------------------------------------------

def ai_move(board):

    print("AI is thinking...")

    best_move = find_best_move(board)

    board[best_move] = "O"

    print(
        f"AI selected position {best_move + 1}."
    )


# ------------------------------------------------------------
# PLAY ONE GAME
# ------------------------------------------------------------

def play_game():

    global human_wins
    global ai_wins
    global draws

    # Empty board
    board = [" "] * 9

    print("\n================================")
    print("       TIC-TAC-TOE")
    print("================================")

    print("\nYou are X")
    print("AI is O")

    display_position_guide()

    display_board(board)

    # Human starts first
    while True:

        # ----------------------------------------------------
        # HUMAN TURN
        # ----------------------------------------------------

        human_move(board)

        display_board(board)

        result = check_winner(board)

        if result is not None:

            if result == "X":

                print("🎉 Congratulations! You won!")
                human_wins += 1

            elif result == "Draw":

                print("🤝 The game is a draw!")
                draws += 1

            break

        # ----------------------------------------------------
        # AI TURN
        # ----------------------------------------------------

        ai_move(board)

        display_board(board)

        result = check_winner(board)

        if result is not None:

            if result == "O":

                print("🤖 AI wins!")
                ai_wins += 1

            elif result == "Draw":

                print("🤝 The game is a draw!")
                draws += 1

            break


# ------------------------------------------------------------
# DISPLAY SCOREBOARD
# ------------------------------------------------------------

def display_scoreboard():

    print("\n================================")
    print("          SCOREBOARD")
    print("================================")

    print(f"Human (X) Wins : {human_wins}")
    print(f"AI (O) Wins    : {ai_wins}")
    print(f"Draws          : {draws}")

    print("================================")


# ------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------

def main():

    print("==============================================")
    print("       TIC-TAC-TOE AI USING MINIMAX")
    print("==============================================")

    print("\nHuman Player : X")
    print("AI Player    : O")

    while True:

        play_game()

        display_scoreboard()

        choice = input(
            "\nDo you want to play again? (y/n): "
        ).lower().strip()

        if choice != "y":

            print("\nThanks for playing Tic-Tac-Toe!")
            print("Final Score:")

            display_scoreboard()

            break


# ------------------------------------------------------------
# START PROGRAM
# ------------------------------------------------------------

if __name__ == "__main__":
    main()