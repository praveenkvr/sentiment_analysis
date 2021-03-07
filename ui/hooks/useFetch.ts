import { useEffect, useState } from 'react';


export default function useFetch(baseUrl: string, options?: RequestInit) {

    const [url, setUrl] = useState(baseUrl);
    const [isFetching, setFetching] = useState(false);
    const [error, setError] = useState(false);
    const [data, setData] = useState(null);

    const doFetch = (updateUrl?: string) => {
        if (updateUrl) {
            setUrl(updateUrl);
        }
        setFetching(true);
        setError(false);
    }

    const fetchData = async () => {
        try {
            const resp = await fetch(url, options);
            const data = await resp.json();
            setFetching(false);
            if (data.error) {
                setError(true);
            } else {
                setData(data);
            }
        } catch (e) {
            console.log(e.stack);
            setFetching(false);
            setError(true);
        }

    };

    useEffect(() => {
        if (isFetching) {
            fetchData();
        }
    }, [isFetching]);

    return { data, isFetching, error, doFetch }
}