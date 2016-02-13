use std::cmp::Ordering;

#[derive(Clone)]
pub struct Rank {
    pub rank: i64,
    pub outputs: [i64; super::bot::nodebrain::TOTAL_OUTPUTS],
}

impl PartialEq for Rank {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank
    }

    fn ne(&self, other: &Self) -> bool {
        self.rank != other.rank
    }
}

impl Eq for Rank {}

impl PartialOrd for Rank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }

    fn lt(&self, other: &Self) -> bool {
        self.rank < other.rank
    }

    fn le(&self, other: &Self) -> bool {
        self.rank <= other.rank
    }

    fn gt(&self, other: &Self) -> bool {
        self.rank > other.rank
    }

    fn ge(&self, other: &Self) -> bool {
        self.rank >= other.rank
    }
}

impl Ord for Rank {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.rank < other.rank {
            Ordering::Less
        } else if self.rank == other.rank {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    }
}
