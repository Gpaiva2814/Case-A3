from core.cleaning import main as cleaning
from core.genre_normalize import main as genre_normalize
from core.review_analysis import main as review_analysis


def main():
    cleaning()
    genre_normalize()
    review_analysis()


if __name__ == "__main__":
    main()
