import React, { ChangeEvent } from 'react';
import css from './Input.module.scss';

type InputProps = {
    onChange: (e: ChangeEvent<HTMLInputElement>) => void;
    value: string;
    onEnter: () => void;
}

const Input = ({ onChange, value, onEnter }: InputProps) => {

    const onKeyUp = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && onEnter) {
            onEnter();
        }
    }

    return (
        <>
            <label className={css['hidden']} htmlFor="inputtext">Query</label>
            <input
                className={css.input}
                id="inputtext"
                placeholder="Search tweets by text or a hashtag"
                aria-label="Search tweets by text or a hashtag"
                type="text" name="input" onChange={onChange} onKeyUp={onKeyUp} value={value} />
        </>
    )
};

export default Input;