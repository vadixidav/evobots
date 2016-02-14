use std::cmp::Ordering;

///Rank lets us rank multiple objects ignoring the data.
///The Ord for Rank is backwards so that BinaryHeap becomes a min heap.
#[derive(Clone)]
pub struct Rank<T> {
    pub rank: i64,
    pub data: T,
}

impl<T> PartialEq for Rank<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank
    }

    fn ne(&self, other: &Self) -> bool {
        self.rank != other.rank
    }
}

impl<T> Eq for Rank<T> {}

impl<T> PartialOrd for Rank<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }

    fn lt(&self, other: &Self) -> bool {
        self.rank > other.rank
    }

    fn le(&self, other: &Self) -> bool {
        self.rank >= other.rank
    }

    fn gt(&self, other: &Self) -> bool {
        self.rank < other.rank
    }

    fn ge(&self, other: &Self) -> bool {
        self.rank <= other.rank
    }
}

impl<T> Ord for Rank<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.rank < other.rank {
            Ordering::Greater
        } else if self.rank == other.rank {
            Ordering::Equal
        } else {
            Ordering::Less
        }
    }
}
