from game import Game

if __name__ == "__main__":
    game = Game(starting_cash=5000)
    game.play(max_hands=300, verbose=True)