"""
Implementation of 2048 game, and strategies to play it.
Allows for alternate board sizes, human player, or three types of computer players.
"""
import random, sys, time, math

WIDTH = 4
HEIGHT = 4
moves = ['i','j','k','m']
board = {}
# move_count counts the nnumber of moves in one game
# group_moves counts the number of moves in a collection of games
move_count = 0
group_moves = 0
# modes: 0 = human player, 1 = random choice, 2 = greedy look-ahead, 3 = genetic
mode = 3
# values for look_ahead and genetic algorithms:
# moves_ahead = number of moves to look ahead
# repeat_moves = how many times to repeat a move to determine its effectiveness
moves_ahead = 8
repeat_moves = 10
# for genetic algorithm:
# num_generate = number of sequences to generate
# num_keep = number to keep after each generation
# num_generations = number of generations to run
num_generate = 40
num_keep = 4
num_generations = 3
# output_modes: 0 = don't show anything at all
# 1 = show only end state and score at end
# 2 = show everything, no pauses
# 3 = show everything, pause 1 second between each move
output_mode = 1
num_games = 5
to_file = False
# path = where to store data if we write to file.
path = ''
score_list = []
start_group = time.time()


def initialize_board(board):
    """
    Function that sets up the initial state of the board
    """
    global fout, move_count, group_moves, start_game
    start_game = time.time()
    # Fill the board with empty squares
    for row_num in range(HEIGHT):
        board[row_num] = []
        for dummy in range(WIDTH):
            board[row_num].append(0)
    # Then add two random number tiles, set score to zero
    add_tile(board)
    add_tile(board)
    score = 0
    # If running a chain of games, add the previous number of moves to the total
    group_moves += move_count
    # and set this game's moves to zero.
    move_count = 0
    # Write game information to file if appropriate
    if to_file and not stop_game:
            filename = path + time.strftime('%Y_%m_%d_%H_%M_%S') + '.txt'
            fout = open(filename, 'w')
            if mode == 0:
                fout.write('Player: Human\n')
            elif mode == 1:
                fout.write('Player: Random Moves\n')
            elif mode == 2:
                fout.write('Player: Greedy Look-Ahead Algorithm\n')
            elif mode == 3:
                fout.write('Player: Genetic Algorithm\n')
            else:
                print('Player Unknown')
                sys.exit(1)
    # And...we're done.
    return board, score

def add_tile(board):
    """
    Function that adds a tile in a random available
    location on the board.  90% of the time, it will
    add a 2, 10% will be a 4.
    """
    available_spots = []
    for row_num in list(board.keys()):
        row = board[row_num]
        for index in range(len(row)):
            if row[index] == 0:
                available_spots.append((row_num,index))
    if len(available_spots) > 0:
        pick_index = random.randrange(len(available_spots))
        pick = available_spots[pick_index]
        row = pick[0]
        index = pick[1]
        choices = [2 for dummy in range(9)]
        choices.append(4)
        to_add = random.choice(choices)
        board[row][index] = to_add


def merge(line, score):
    """
    Function that merges a single row or column in 2048.
    Default dir = left
    """
    line_2 = []
    for dummy_index in range(len(line)):
        line_2.append(0)
    index = 0
    previous = 0
    for entry in line:
        if not entry == 0 and not entry == previous:
            line_2[index] = entry
            index += 1
            previous = entry
        elif not entry == 0 and entry == previous:
            line_2[index-1] = 2*entry
            score += 2*entry
            previous = 0
    changed = False
    for index in range(len(line)):
        if line[index] != line_2[index]:
            changed = True
    return line_2, changed, score

def rows_to_cols(board, reversed = False):
    """
    Helper function to convert rows to columns
    """
    new_board = {}
    # The following is only necessary if you allow for WIDTH != HEIGHT
    if reversed:
        width = HEIGHT
    else:
        width = WIDTH
    for index in range(width):
        new_row = []
        for key in list(board.keys()):
            new_row.append(board[key][index])
        new_board[index] = new_row
    return new_board

def merge_left(board, score, check = False):
    """
    Function that merges all rows left
    """
    new_board = {}
    board_changed = False
    for key in list(board.keys()):
        new_board[key], row_changed, score = merge(board[key], score)
        if row_changed:
            board_changed = True
        if check:
            print(new_board[key], row_changed, end=' ')
    if check:
        print(board_changed)
    return new_board, board_changed, score

def merge_right(board, score, check = False):
    """
    Function that merges all rows right
    """
    new_board = {}
    board_changed = False
    for key in list(board.keys()):
        board[key].reverse()
        new_row, row_changed, score = merge(board[key], score)
        new_row.reverse()
        new_board[key] = new_row
        if row_changed:
            board_changed = True
        if check:
            print(new_row, row_changed, end=' ')
    if check:
        print(board_changed)
    return new_board, board_changed, score

def merge_up(board, score, check=False):
    """
    Function that merges all columns up
    """
    new_board, changed, score = merge_left(rows_to_cols(board), score, check)
    new_board = rows_to_cols(new_board, True)
    return new_board, changed, score

def merge_down(board, score, check=False):
    """
    Function that merges all columns down
    """
    new_board, changed, score = merge_right(rows_to_cols(board), score, check)
    new_board = rows_to_cols(new_board, True)
    return new_board, changed, score

def get_move(board, score):
    """
    Function that inputs the move and calls the correct directional merge
    """
    global move_count, stop_game
    if mode == 0:
        print()
        print('j=left, k=right, i=up, m=down, q=quit, then hit <Enter>')
        dir = input ('What is your move? ')
    if mode == 1:
        dir = random.choice(('j','m','k','i'))
    if mode == 2:
        dir = look_ahead(board, score)
    if mode == 3:
        dir = genetic(board, score)
    if dir == 'j':
        new_board, changed, score = merge_left(board, score)
        if mode > 0 and output_mode > 1:
            print('Next Move: <--')
    elif dir == 'k':
        new_board, changed, score = merge_right(board, score)
        if mode > 0 and output_mode > 1:
            print('Next Move: -->')
    elif dir == 'i':
        new_board, changed, score = merge_up(board, score)
        if mode > 0 and output_mode > 1:
            print('Next Move: ^')
    elif dir == 'm':
        new_board, changed, score = merge_down(board, score)
        if mode > 0 and output_mode > 1:
            print('Next Move: v')
    elif dir == 'q':
        print('Exiting...')
        new_board = board
        changed = False
        stop_game = True
    else:
        print('Move not understood. Please try again.')
        new_board = board
        changed = False
    if changed:
        move_count += 1
        add_tile(new_board)
        if check_for_end(new_board):
            if output_mode > 0:
                print()
                print()
                for row in list(new_board.values()):
                    print(printable_row(row))
            if to_file:
                fout.write('Move: ' + str(move_count) + ' Score: ' + str(score) + '\n')
                for row in list(new_board.values()):
                    fout.write(printable_row(row) + '\n')
                end_game = time.time()
                fout.write ('Time of game: ' + str(end_game - start_game) + '\n')
                fout.close()
            if mode == 0:
                print('Game Over. Final Score:', score)
                stop_game = True
            if mode > 0:
                if output_mode > 0:
                    print(score)
                score_list.append(score)
                new_board, score = initialize_board(new_board)
    return new_board, score

def printable_row(row):
    """
    Function to print a row on screen
    """
    row_string = ''
    for entry in row:
        if entry > 0:
            row_string += '%(number)8d' % {'number': entry}
        else:
            row_string += '       .'
    return row_string

def check_for_end(board):
    """
    Function to see if game is over
    """
    for row in list(board.values()):
        for entry in row:
            if entry == 0:
                return False
    changed = False
    test_board = {}
    for key in list(board.keys()):
        test_board[key] = []
        for entry in board[key]:
            test_board[key].append(entry)
    test_board, changed, dummy = merge_left(test_board, 0)
    if changed:
        return False
    test_board, changed, dummy = merge_right(test_board, 0)
    if changed:
        return False
    test_board, changed, dummy = merge_up(test_board, 0)
    if changed:
        return False
    test_board, changed, dummy = merge_down(test_board, 0)
    if changed:
        return False
    else:
        return True

def look_ahead(board, score):
    """
    Function to implement a greedy look-ahead algorithm to decide what move to make
    """
    global stop_game
    combinations = {}
    for index in range(4):
        combinations[index] = [moves[index]]
    for num in range(moves_ahead-1):
        tmp_combs = {}
        for key in list(combinations.keys()):
            for index in range(4):
                tmp_combs[4*key + index] = []
                for entry in combinations[key]:
                    tmp_combs[4*key + index].append(entry)
                tmp_combs[4*key + index].append(moves[index])
        for key in list(tmp_combs.keys()):
            combinations[key] = tmp_combs[key]
    scores = {}
    for key in list(combinations.keys()):
        scores[key] = []
        for dummy in range(repeat_moves):
            tmp_score = score
            tmp_board = {}
            for key in list(board.keys()):
                tmp_board[key] = []
                for tile in board[key]:
                    tmp_board[key].append(tile)
            print(board, tmp_board)
            for entry in combinations[key]:
                if entry == 'j':
                    tmp_board, changed, tmp_score = merge_left(tmp_board, tmp_score)
                if entry == 'k':
                    tmp_board, changed, tmp_score = merge_right(tmp_board, tmp_score)
                if entry == 'i':
                    tmp_board, changed, tmp_score = merge_up(tmp_board, tmp_score)
                if entry == 'm':
                    tmp_board, changed, tmp_score = merge_down(tmp_board, tmp_score)
                if not check_for_end(tmp_board):
                    add_tile(tmp_board)
            scores[key].append(tmp_score - score)
    avg_scores = {}
    for key in list(scores.keys()):
        avg_scores[key] = sum(scores[key])//len(scores[key])
    max = 0
    for key in list(avg_scores.keys()):
        if avg_scores[key] > max:
            max = avg_scores[key]
    options = []
    for key in list(avg_scores.keys()):
        if avg_scores[key] == max:
            options.append(key)
    dirs = []
    move_num = 0
    while len(dirs) == 0 and move_num < moves_ahead:
        dirs = []
        for key in options:
            dirs.append(combinations[key][move_num])
        tmp_dirs = []
        for entry in dirs:
            tmp_board = {}
            for key in list(board.keys()):
                tmp_board[key] = []
                for tile in board[key]:
                    tmp_board[key].append(tile)
            if entry == 'j':
                tmp_board, changed, tmp_score = merge_left(tmp_board, tmp_score)
            elif entry == 'k':
                tmp_board, changed, tmp_score = merge_right(tmp_board, tmp_score)
            elif entry == 'i':
                tmp_board, changed, tmp_score = merge_up(tmp_board, tmp_score)
            elif entry == 'm':
                tmp_board, changed, tmp_score = merge_down(tmp_board, tmp_score)
            if changed:
                tmp_dirs.append(entry)
        dirs = []
        for entry in tmp_dirs:
            dirs.append(entry)
        move_num += 1
    if dirs:
        dir = random.choice(dirs)
    else:
        dir = random.choice(moves)
    return dir

def generate_random_move_list (length):
    """
    Function to generate a random sequence of moves of given length for the
    genetic algorithm. Returns a list with the sequence of moves.
    """
    move_list = []
    for dummy in range(length):
        move_list.append(random.choice(moves))
    return move_list

def modify_move_list (move_list):
    """
    Function to modify an existing move list for the genetic algorithm.  Returns
    a list of moves of the same length with at least one move randomly changed.
    """
    length = len(move_list)
    new_moves = []
    for move in move_list:
        new_moves.append(move)
    num_to_change = random.randrange(1, length - 1)
    locs_to_change = [0 for dummy in range(length)]
    for num_changed in range(num_to_change):
        loc_to_change = random.randrange(length - num_changed)
        index = 0
        for dummy in range(loc_to_change):
            while locs_to_change[index] == 1:
                index += 1
            index += 1
        while locs_to_change[index] == 1:
            index += 1
        locs_to_change[index] = 1
    for index in range(length):
        if locs_to_change[index] == 1:
            new_move = move_list[index]
            while new_move == move_list[index]:
                new_move = random.choice(moves)
            new_moves[index] = new_move
    return new_moves

def copy_board (board):
    """
    Function to return an independent copy of the board to use to test a list
    of moves without changing the original board.
    """
    tmp_board = {}
    for key in list(board.keys()):
        tmp_board[key] = []
        for tile in board[key]:
            tmp_board[key].append(tile)
    return tmp_board

def execute_move(move, board, score):
    """
    Function to carry out on a move on the board given, and record whether or
    not the board has changed, and what the new score is.
    """
    if move == 'j':
        board, changed, score = merge_left(board, score)
    elif move == 'k':
        board, changed, score = merge_right(board, score)
    elif move == 'i':
        board, changed, score = merge_up(board, score)
    elif move == 'm':
        board, changed, score = merge_down(board, score)
    if changed:
        add_tile(board)
    return board, changed, score

def score_moves (board, score, move_list):
    """
    Function to run a move list several times and return the average score
    achieved by this lsist of moves.
    """
    scores = []
    for dummy in range(repeat_moves):
        tmp_board = copy_board(board)
        tmp_score = score
        changed = False
        for move in move_list:
            tmp_board, move_changed, tmp_score = execute_move(move, tmp_board, tmp_score)
            if move_changed:
                changed = True
        scores.append(tmp_score)
    avg_score = 0
    for tmp_score in scores:
        avg_score += (tmp_score - score)
    avg_score /= repeat_moves
    return avg_score

def lists_to_keep(scores):
    """
    Function to return the top-scoring move lists
    """
    top_scores = [0 for dummy in range(num_keep)]
    to_keep = [-1 for dummy in range(num_keep)]
    for score_index in range(len(scores)):
        for kept_score in top_scores:
            if scores[score_index] > kept_score:
                kept_index = top_scores.index(kept_score)
                top_scores.insert(kept_index, scores[score_index])
                to_keep.insert(kept_index, score_index)
                top_scores.pop()
                to_keep.pop()
                break
    to_keep.sort()
    return to_keep


def genetic (board, score):
    """
    Function to implement a genetic algorithm for finding the best moves
    """
    move_lists = []
    for dummy in range(num_generate):
        move_lists.append(generate_random_move_list(moves_ahead))
    for generation in range(num_generations):
        scores = []
        for move_list in move_lists:
            scores.append(score_moves(board, score, move_list))
        to_keep = lists_to_keep(scores)
        for index in range(num_keep):
            move_lists[index] = move_lists[to_keep[index]]
        move_lists = move_lists[0:num_keep]
        variations = (num_generate // num_keep) - 1
        for list_num in range(num_keep):
            for dummy in range(variations):
                list_to_add = modify_move_list(move_lists[list_num])
                move_lists.append(list_to_add)
    scores = []
    for move_list in move_lists:
        score = score_moves(board, score, move_list)
        scores.append(score)
    max_score = 0
    for score in scores:
        if score > max_score:
            max_score = score
    tmp_dirs = []
    for index in range(len(scores)):
        if scores[index] == max_score:
            tmp_dirs.append(move_lists[index][0])
    dirs_to_kill = []
    for dir in tmp_dirs:
        tmp_board = copy_board(board)
        tmp_board, changed, tmp_score = execute_move(dir, tmp_board, score)
        if changed:
            dirs_to_kill.append(False)
        else:
            dirs_to_kill.append(True)
    dirs = []
    for index in range(len(dirs_to_kill)):
        if not dirs_to_kill[index]:
            dirs.append(tmp_dirs[index])
    if dirs:
        dir = random.choice(dirs)
    else:
        dir = random.choice(moves)
    return dir

stop_game = False
board, score = initialize_board(board)
while not stop_game:
    if move_count == 0:
        if mode == 2:
            print('Look Ahead: ' + str(moves_ahead))
            print(' Repeat Moves: ' + str(repeat_moves))
        if mode == 2 and to_file:
            fout.write ('Look Ahead: ' + str(moves_ahead) + ' Repeat Moves: '
                + str(repeat_moves) + '\n')
        if mode == 3:
            print('Genetic, with Look Ahead: ' + str(moves_ahead))
            print('Repeat Moves: ' + str(repeat_moves))
            print('Generations: ' + str(num_generations))
            print('Sequences Generated per generation: ' + str(num_generate))
            print('Sequences to Keep per generation: ' + str(num_keep))
        if mode == 3 and to_file:
            fout.write ('Genetic, with Look Ahead: ' + str(moves_ahead) + ' Repeat Moves: '
                + str(repeat_moves) + ' Generations: ' + str(num_generations) +
                ' Sequences Generated per generation: ' + str(num_generate) +
                ' Sequences to Keep per generation: ' + str(num_keep) + '\n')
    if output_mode > 1:
        print('------------------------')
        print('Move #:', move_count)
        print('Score:', score)
        print()
        for row in list(board.values()):
            print(printable_row(row))
        print()
    if to_file:
        fout.write('Move: ' + str(move_count) + ' Score: ' + str(score) + '\n')
        for row in list(board.values()):
            fout.write(printable_row(row) + '\n')
    board, score = get_move(board, score)
    if output_mode > 2:
        time.sleep(1)
    if mode > 0 and len(score_list) >= num_games:
        stop_game = True
if len(score_list) > 1:
    length = len(score_list)
    sum = 0
    max = 0
    min = 1000
    for entry in score_list:
        sum += entry
        if entry < min:
            min = entry
        if entry > max:
            max = entry
    average = sum/length
    sd_sum = 0
    for entry in score_list:
        sd_sum += (average - entry)**2
    sd = math.sqrt(sd_sum/(length-1))
    sorted_scores = sorted(score_list)
    if length % 2 == 0:
        med = (sorted_scores[length//2 - 1] + sorted_scores[length//2])/2
    else:
        med = sorted_scores[length//2]
    if length % 4 == 0:
        first_q = (sorted_scores[length//4 - 1] + sorted_scores[length//4])/2
        third_q = (sorted_scores[3*(length//4) - 1] + sorted_scores[3*(length//4)])/2
    if length % 4 == 1:
        first_q = sorted_scores[length//4]
        third_q = sorted_scores[3*(length//4)]
    if length % 4 == 2:
        first_q = sorted_scores[length//4]
        third_q = sorted_scores[3*(length//4) + 1]
    if length % 4 == 3:
        first_q = (sorted_scores[length//4] + sorted_scores[length//4 + 1])/2
        third_q = (sorted_scores[3*(length//4) + 1] + sorted_scores[3*(length//4) + 2])/2
    print('Average:', average, 'Sd:', round(sd,2))
    print('Min:', min, 'Q1:', first_q, 'Med:', med, 'Q3:', third_q, 'Max:', max)
    print('Total Moves:', group_moves)
    end_group = time.time()
    print('Total time taken: ', end_group - start_group, 'seconds')
    print('Moves per second:', group_moves / (end_group - start_group))
