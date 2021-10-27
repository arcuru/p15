//#![feature(test)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

#[allow(dead_code)]
#[derive(PartialEq, Eq, Clone, Copy)]
enum Move {
    Right,
    Left,
    Up,
    Down,
}

#[allow(dead_code)]
#[derive(PartialEq, Eq, Clone)]
struct P15 {
    // The board info
    pub(crate) board: [[u8; 4]; 4],
}

#[allow(dead_code)]
#[derive(Eq, Clone)]
struct P15Solver {
    // The board info
    board: P15,
    moves: Vec<Move>,
    pub(crate) distance: usize,
}

impl Ord for P15Solver {
    fn cmp(&self, other: &Self) -> Ordering {
        other.value().cmp(&self.value())
    }
}

impl PartialOrd for P15Solver {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for P15Solver {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

#[allow(dead_code)]
impl P15Solver {
    pub fn new(original: &P15) -> P15Solver {
        P15Solver {
            board: original.clone(),
            moves: Vec::new(),
            distance: original.manhattan_distance(),
        }
    }

    pub fn value(&self) -> usize {
        self.distance + self.moves.len()
    }

    pub fn add_next_steps(&self, heap: &mut BinaryHeap<P15Solver>) {
        let zero = self.board.zero();
        if zero.0 != 0 && (self.moves.len() == 0 || *self.moves.last().unwrap() != Move::Down) {
            let mut c = self.clone();
            c.board.swap(zero, (zero.0 - 1, zero.1));
            c.moves.push(Move::Up);
            c.distance = c.board.manhattan_distance();
            heap.push(c);
        }
        if zero.0 != 3 && (self.moves.len() == 0 || *self.moves.last().unwrap() != Move::Up) {
            let mut c = self.clone();
            c.board.swap(zero, (zero.0 + 1, zero.1));
            c.moves.push(Move::Down);
            c.distance = c.board.manhattan_distance();
            heap.push(c);
        }
        if zero.1 != 0 && (self.moves.len() == 0 || *self.moves.last().unwrap() != Move::Right) {
            let mut c = self.clone();
            c.board.swap(zero, (zero.0, zero.1 - 1));
            c.moves.push(Move::Left);
            c.distance = c.board.manhattan_distance();
            heap.push(c);
        }
        if zero.1 != 3 && (self.moves.len() == 0 || *self.moves.last().unwrap() != Move::Left) {
            let mut c = self.clone();
            c.board.swap(zero, (zero.0, zero.1 + 1));
            c.moves.push(Move::Right);
            c.distance = c.board.manhattan_distance();
            heap.push(c);
        }
    }
}

#[allow(dead_code)]
impl P15 {
    pub fn new(board: [[u8; 4]; 4]) -> P15 {
        P15 { board }
    }

    pub fn new_solved() -> P15 {
        P15::new([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]])
    }

    /// Accesses a value at the given row and column
    pub fn at(&self, row: usize, column: usize) -> u8 {
        self.board[row][column]
    }

    /// Sets the value of a board location
    pub fn set(&mut self, row: usize, column: usize, value: u8) {
        self.board[row][column] = value;
    }

    pub(crate) fn swap(&mut self, a: (usize, usize), b: (usize, usize)) {
        let tmp = self.at(a.0, a.1);
        self.set(a.0, a.1, self.at(b.0, b.1));
        self.set(b.0, b.1, tmp);
    }

    fn idx(&self, idx: usize) -> u8 {
        self.at(idx / 4, idx % 4)
    }

    /// Get the coordinates of the empty space.
    pub fn zero(&self) -> (usize, usize) {
        for i in 0..4 {
            for j in 0..4 {
                if self.at(i, j) == 0 {
                    return (i, j);
                }
            }
        }
        panic!();
    }

    /// Checks whether the board is a valid board containing only the numbers 0-15
    pub fn is_valid(&self) -> bool {
        let mut mask: u16 = 0;
        for i in &self.board {
            for j in i {
                if *j > 15 as u8 {
                    return false;
                }
                mask |= (1 as u16) << j;
            }
        }
        mask == 0xFFFF
    }

    /// Count inversions
    fn count_inversions(&self) -> usize {
        let mut inversions = 0;
        for i in 0..15 {
            if self.idx(i) == 0 {
                continue;
            }
            for j in (i + 1)..16 {
                if self.idx(j) == 0 {
                    continue;
                }
                if self.idx(i) > self.idx(j) {
                    inversions += 1;
                }
            }
        }
        inversions
    }

    /// Return the row that contains the empty space
    fn empty_row(&self) -> usize {
        self.zero().0
    }

    /// Checks to see if the board is solvable
    pub fn is_solvable(&self) -> bool {
        // Rules taken from: https://www.geeksforgeeks.org/check-instance-15-puzzle-solvable/
        // The code is my own

        // In general, for a given grid of width N, we can find out check if a N*N – 1 puzzle is solvable or not by following below simple rules :
        // 1. If N is odd, then puzzle instance is solvable if number of inversions is even in the input state.
        // 2. If N is even, puzzle instance is solvable if
        //      the blank is on an even row counting from the bottom (second-last, fourth-last, etc.) and number of inversions is odd.
        //      the blank is on an odd row counting from the bottom (last, third-last, fifth-last, etc.) and number of inversions is even.
        // 3. For all other cases, the puzzle instance is not solvable.

        if !self.is_valid() {
            return false;
        }

        let inversions = self.count_inversions();
        if self.empty_row() & 1 == 1 {
            inversions & 1 == 0
        } else {
            inversions & 1 == 1
        }
    }

    fn single_distance(&self, row: usize, col: usize) -> usize {
        let mut dist = 0;
        // Don't count the blank square
        let val = self.at(row, col) as i32;
        dist += ((row as i32) - (val - 1) / 4).abs();
        dist += ((col as i32) - (val - 1) % 4).abs();
        dist as usize
    }

    /// Calculates the manhattan distance for all the tiles. This is a lower bound on the number of moves required to solve the puzzle
    fn manhattan_distance(&self) -> usize {
        let mut dist = 0;
        for i in 0..4 {
            for j in 0..4 {
                // Don't count the blank square
                if self.at(i, j) != 0 {
                    dist += self.single_distance(i, j);
                }
            }
        }
        dist as usize
    }
}

// Returns the solution to this puzzle. Caches the resulting solution in the struct for future reference.
#[allow(dead_code)]
fn solve(board: &P15) -> Result<Vec<Move>, ()> {
    if !board.is_solvable() {
        return Err(());
    }

    // Create the initial solver object and add to the heap.
    let mut heap = BinaryHeap::new();
    heap.push(P15Solver::new(board));

    let solution_distance = usize::MAX;
    loop {
        let x = heap.pop().unwrap();
        if x.value() > solution_distance {
            break;
        }
        if x.distance == 0 {
            //solution_distance = x.value();
            // FIXME: This only returns 1 solution
            return Ok(x.moves);
        }

        x.add_next_steps(&mut heap);
    }
    Err(())
}

impl fmt::Display for P15 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "+----+----+----+----+")?;
        for i in 0..4 {
            for j in 0..4 {
                if self.at(i, j) != 0 {
                    write!(f, "| {:2} ", self.at(i, j))?;
                } else {
                    write!(f, "|    ")?;
                }
            }
            writeln!(f, "|")?;
            writeln!(f, "+----+----+----+----+")?;
        }
        Ok(())
    }
}

#[allow(dead_code)]
fn vec_to_string(v: &Vec<Move>) -> String {
    let mut s = String::new();
    for x in v {
        match x {
            Move::Left => s.push('l'),
            Move::Right => s.push('r'),
            Move::Up => s.push('u'),
            Move::Down => s.push('d'),
        }
    }
    s
}

fn main() {
    //println!("{}", P15::new_solved());
    println!("{}", solve(&P15::new([[1, 10, 15, 4], [13, 6, 3, 8], [2, 9, 12, 7], [14, 5, 0, 11]])).unwrap().len());
    /*
    let l = solve(&P15::new([
            [15, 14, 1, 6],
            [9, 11, 4, 12],
            [0, 10, 7, 3],
            [13, 8, 5, 2]
        ]))
        .unwrap()
        .len();
    println!("{}",l);
    */
}

#[cfg(test)]
#[allow(dead_code)]
mod tests {
    use super::*;

    extern crate test;
    use test::Bencher;

    static ROSETTA_EXAMPLE: [[u8; 4]; 4] =
        [[15, 14, 1, 6], [9, 11, 4, 12], [0, 10, 7, 3], [13, 8, 5, 2]];
    static ROSETTA_SOLUTION1: &str = "rrrulddluuuldrurdddrullulurrrddldluurddlulurruldrdrd";
    static ROSETTA_SOLUTION2: &str = "rrruldluuldrurdddluulurrrdlddruldluurddlulurruldrrdd";

    #[test]
    fn default_is_valid() {
        let mut tmp = P15::new_solved();
        assert_eq!(true, tmp.is_valid());
        tmp.set(0, 0, 5);
        assert_eq!(false, tmp.is_valid());
    }

    #[test]
    fn default_is_solvable() {
        assert_eq!(true, P15::new_solved().is_solvable());
    }

    #[test]
    fn inverted_is_unsolvable() {
        let mut inv = P15::new_solved();
        inv.set(0, 0, 2);
        inv.set(0, 1, 1);
        assert_eq!(true, inv.is_valid());
        assert_eq!(false, inv.is_solvable());
    }

    #[test]
    fn one_move_solvable() {
        let mut b = P15::new_solved();
        b.set(3, 3, 12);
        b.set(2, 3, 0);
        assert_eq!(true, b.is_valid());
        assert_eq!(true, b.is_solvable());
    }

    #[test]
    fn manhattan_distance_of_solved() {
        assert_eq!(0, P15::new_solved().manhattan_distance());
        let mut inv = P15::new_solved();
        inv.set(0, 0, 2);
        inv.set(0, 1, 1);
        assert_eq!(2, inv.manhattan_distance());
        //assert_eq!(0, P15::new(ROSETTA_EXAMPLE).manhattan_distance());
    }

    #[test]
    fn zero_point() {
        assert_eq!(P15::new_solved().zero(), (3, 3));
        assert_eq!(P15::new(ROSETTA_EXAMPLE).zero(), (2, 0));
    }

    #[test]
    fn one_off() {
        let mut b = P15::new_solved();
        b.set(3, 3, 12);
        b.set(2, 3, 0);
        assert_eq!(vec_to_string(&solve(&b).unwrap()), "d");
    }

    #[test]
    fn magic_square() {
        assert_eq!(
            solve(&P15::new([
                [1, 10, 15, 4],
                [13, 6, 3, 8],
                [2, 9, 12, 7],
                [14, 5, 0, 11]
            ]))
            .unwrap()
            .len(),
            35
        );
    }

    #[bench]
    fn magic_square_bench(b: &mut Bencher) {
        let board = P15::new([[1, 10, 15, 4], [13, 6, 3, 8], [2, 9, 12, 7], [14, 5, 0, 11]]);
        b.iter(|| solve(&board));
    }

    /*
    #[bench]
    fn rosetta_bench(b: &mut Bencher) {
        let board = P15::new(ROSETTA_EXAMPLE);
        b.iter(|| solve(&board));
    }
    */
}
