#!/bin/bash

# Download all CSV files listed below into the current directory
# Uses wget and provides error handling for each file

set -e

links=(
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Movies_and_TV.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_CDs_and_Vinyl.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Clothing_Shoes_and_Jewelry.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Home_and_Kitchen.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Kindle_Store.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Sports_and_Outdoors.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Cell_Phones_and_Accessories.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Health_and_Personal_Care.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Toys_and_Games.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Video_Games.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Tools_and_Home_Improvement.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Apps_for_Android.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Office_Products.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Pet_Supplies.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Automotive.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Grocery_and_Gourmet_Food.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Patio_Lawn_and_Garden.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Baby.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Digital_Music.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Musical_Instruments.csv"
	"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Amazon_Instant_Video.csv"
)

for url in "${links[@]}"; do
	fname="$(basename "$url")"
	echo "Downloading $fname ..."
	if wget -q --show-progress -O "$fname" "$url"; then
		echo "Successfully downloaded $fname"
	else
		echo "Error downloading $fname from $url" >&2
		exit 1
	fi
done
