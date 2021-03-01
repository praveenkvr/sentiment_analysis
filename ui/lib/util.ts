
export type Tweet = {
    isPositive: boolean;
    result: number;
    text: string
}

type Results = {
    results: {
        percentage: number, tweets: Tweet[]
    }
}

export function processResults(data: Results | null) {

    if (data && data.results) {
        const { results: { percentage = undefined, tweets = [] } = {} } = data;
        return {
            percentage, tweets
        }
    }

    return null;
}

export function getTopPositiveTweets(tweets: Tweet[] = [], max = 3) {
    const positive = tweets.filter(tweet => tweet.isPositive).sort((a, b) => b.result - a.result);
    if (positive.length > max) {
        return positive.slice(0, max);
    }
    return positive;
}

export function getTopNegativeTweets(tweets: Tweet[] = [], max = 3) {
    const negative = tweets.filter(tweet => !tweet.isPositive).sort((a, b) => a.result - b.result);
    if (negative.length > max) {
        return negative.slice(0, max);
    }
    return negative;
}