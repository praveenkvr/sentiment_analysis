import React from 'react';
import css from './Button.module.scss';

type ButtonProps = {
    onClick: (e: React.MouseEvent<HTMLButtonElement>) => void;
    children: string;
}

const Input = ({ children, onClick }: ButtonProps) => {

    return (
        <button className={css.button} type="button" onClick={onClick}>{children}</button>
    )
};

export default Input;