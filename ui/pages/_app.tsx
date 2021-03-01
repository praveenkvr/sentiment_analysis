import type { AppProps } from 'next/app';
import 'normalize.css/normalize.css';
import '../styles/style.css';

export default function MyApp({ Component, pageProps }: AppProps) {
    return <Component {...pageProps} />
}