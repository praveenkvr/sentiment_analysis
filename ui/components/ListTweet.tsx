import type { Tweet } from '../lib/util';
import css from './ListTweet.module.scss';

type Porps = {
    tweets: Tweet[] | undefined
}

export default function ListTweet({ tweets }: Porps) {
    if (!tweets) {
        return null;
    }

    return (
        <ul className={css['list-container']}>
            {tweets.map(tweet =>
                <li key={tweet.result} className={css['list-item']}>{tweet.text}</li>
            )}
        </ul>
    )
}