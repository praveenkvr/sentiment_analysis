import React from 'react';
import css from './Card.module.scss';

type Props = {
    header: string;
    children: React.ReactNode
}

export default function Card({ children, header }: Props) {

    return (
        <section className={css.card}>
            <header><h2>{header}</h2></header>
            <main className={css['card-content']}>
                {children}
            </main>
        </section>
    )
}