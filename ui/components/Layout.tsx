import React, { ReactNode } from 'react'
import Head from 'next/head'
import css from './layout.module.scss';

type Props = {
  children?: ReactNode
  title?: string
}

const Layout = ({ children, title = 'Sentiment Analysis' }: Props) => (
  <div className={css.container}>
    <Head>
      <title>{title}</title>
      <meta charSet="utf-8" />
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
