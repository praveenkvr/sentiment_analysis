import React from 'react';

export type GridItemProps = { children: React.ReactNode | string; }

export default function GridItem(props: GridItemProps) {
    return <div className="grid-item">
        {props.children}
    </div>
}