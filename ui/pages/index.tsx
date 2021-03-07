
import Layout from '../components/Layout'
import Input from '../components/Input';
import Button from '../components/Button';
import Grid from '../components/Grid';
import Card from '../components/Card';
import ListTweet from '../components/ListTweet';
import useFetch from '../hooks/useFetch';

import { ChangeEvent, useEffect, useState } from 'react';
import { processResults, getTopPositiveTweets, getTopNegativeTweets } from '../lib/util';

import css from '../styles/index.module.scss';

const BASE_URL = `/searchandanalyze`;

const IndexPage = () => {

  const [value, setValue] = useState('');
  const [buttonText, setButtonText] = useState('Analyze');
  const { data, error, doFetch, isFetching } = useFetch(BASE_URL, {
    mode: 'cors'
  })

  useEffect(() => {
    if (!isFetching) {
      setButtonText('Analyze');
    }
  }, [isFetching]);

  const onClick = () => {

    if (!value) {
      alert('input field required')
    }

    setButtonText('Analyzing...');
    const url = new URL(BASE_URL);
    const params = { q: value }
    Object.keys(params).forEach(key => url.searchParams.append(key, value))
    doFetch(url.toString());
  }

  const results = processResults(data);
  const isPositive = results?.percentage && results.percentage > 0.5;
  const percentage = `${results?.percentage && Math.floor(results.percentage * 100)} %`;
  return (
    <Layout title="Home">
      <div>
        <h1 className={css['large-text']}>Sentiment Analysis</h1>
      </div>
      <div className={css['input-elements']}>
        <Input onChange={(e: ChangeEvent<HTMLInputElement>) => setValue(e.target?.value || '')} value={value} onEnter={onClick} />
        <Button onClick={onClick}>{buttonText}</Button>
      </div>
      <div className={css['inputs-spacer']}></div>
      {!isFetching && results &&
        <div className="results">
          <Grid>
            <Grid.GridItem><Card header={'Average'}><p className={isPositive ? css['percentage-text-positive'] : css['percentage-text-negative']}>{percentage}</p></Card></Grid.GridItem>
            <Grid.GridItem><Card header={'Top Positive'}><ListTweet tweets={getTopPositiveTweets(results.tweets)} /></Card></Grid.GridItem>
            <Grid.GridItem><Card header={'Top Negative'}><ListTweet tweets={getTopNegativeTweets(results.tweets)} /></Card></Grid.GridItem>
          </Grid>
        </div>
      }
      {error &&
        <div className="error">{"Failed to get data"}</div>
      }
    </Layout>
  )
}

export default IndexPage
