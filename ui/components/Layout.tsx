import React, { ReactNode } from 'react'
import Head from 'next/head'
import css from './Layout.module.scss';

type Props = {
  children?: ReactNode
  title?: string
}

const Layout = ({ children, title = 'Sentiment Analysis' }: Props) => (
  <div className={css.container}>
    <Head>
      <title>{title}</title>
      <meta name="description" content="Analyze tweets sentiment based on search" />
      <meta name="viewport" content="initial-scale=1.0, width=device-width" />
      <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png" />
      <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png" />      <meta charSet="utf-8" />
      <meta name="viewport" content="initial-scale=1.0, width=device-width" />
    </Head>
    <header>
    </header>
    {children}
    <footer>
    </footer>
  </div>
)

export default Layout