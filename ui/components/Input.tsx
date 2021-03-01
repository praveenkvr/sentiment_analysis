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
        <input className={css.input} placeholder="Enter a string or a hashtag" type="text" name="input" id="input" onChange={onChange} onKeyUp={onKeyUp} value={value} />
    )
};

export default Input;