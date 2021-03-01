import React from 'react';
import GridItem, { GridItemProps } from './GridItem';
import css from './grid.module.scss';

type Props = {
    columns?: number;
    children: React.ReactElement<GridItemProps>[] | React.ReactElement<GridItemProps>
}

export default class Grid extends React.Component<Props> {

    public static GridItem = GridItem;

    render() {

        const { columns } = this.props;
        const styles = columns ? { gridTemplateColumns: `repeat(${columns}, 1fr)` } : {}

        return (
            <div className={css['grid-container']} style={styles}>
                {this.props.children}
            </div>
        )
    }
}