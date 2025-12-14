The aim of the pseudocode is to create a poker game which has two players. The aim is to run a simulation on a poker game.:

There would be a game class which would just initialize the 2 players.

Class Game:
	Def init:
		Player1: AgentClass(starting_cash : int) <- hardcode that to 10 for now 
		Player2: AgentClass(starting_cash : int) <- hardcode that to 10 for now 


there would be the class Board, which would store the cards which are on the table and whether the cards is opened or closed

Class Board:
	Def init: 
		Cards, ( defines all 5 cards on the table)
		OpenIndexes (defines whether the card has been opened yet or not)

	Def get_board:
		this will return the list of cards which are open/closed, for example: (card1, card2, card3, X, X) where X will denote the cards which are hidden. 

There would be the function next_hand(), this would be in charge of assigning the cards for player 1 player 2, and the board cards. The function would assign cards like 2S for 2 of spades, etc. to all the players but it would make sure they aren’t repeated. Also make sure all cards are randomized every single time (including running the same program twice. 

Def next_hand()
	player1.init(card1, card2, starting_cash) <- the starting_cash would have changed based on whether they won or lost
	

There would be another function which mentions how much money each player is betting. It would store a list of actions in BetState = [(player1, raise(10)), (player2, call),(player1, fold)], this is an example.

Def bet():
		bs : BetState = []

		Do:  Make sure the first action is a raise of 1.
			betstate.Append(Player1.get_action()) <- this is going to add the action which the player1 chooses to take and adds that to the list. In case it is a fold then the process will stop right there.
			get p2_action(BetState)


		While:
			At least 1 person puts money in or no one folds

	Return the entire list betstate. Also return the amount of money entering the pot.

The part where the actual game would occur is the below function. It would be in charge of which hand is being played, and calling the other functions.

Def Play():
	While player1 or player2 starting_cash !=0 and next_hand():
	
		For round=0 till 3:
			It would call the board class and change the openindex.  It would change, in round 0 where no cards would be seen, and then round 1 three cards would be seen, and then there would be round 2 where 1 more card would be seen and round 3 where all cards would be seen.
After the board is defined it would call the bet function defined before

bet()

The bet amount would get added into the pot

If this is the final round, and once all bets are placed. The showdown function is called which will check based on all the cards available which player won.
Based on which player won the pot amount will be added into the starting_amount of the player.


	
Create a function for showdown, to check which player won. The goal is to see all combinations of possible hands and see which one is the greatest.


There would be a player function, which would define the cards and the starting_money which it has.

Class player:
	Def init:
		cards, starting_amount
		Also make sure it stores a history of all hands and history of all bets.
	Get_action:
		Call (no amount would be deducted or added, unless the other player raised then it would match the same amount and call the raise)
		Raise: this would deduct from the total starting_amount which the player has, and it cannot go negative. The amount raised would go in the pot.

		Fold: would fold.

		Would be defaulted to always choose the call option.